import copy
from typing import Any, Dict, Tuple

import torch
import numpy as np
from torch_geometric.data import Data
from monai.transforms import Transform
from sklearn.metrics.pairwise import haversine_distances
from pytorch_lightning.utilities import move_data_to_device

from src.utils.geometry import IcoSphere, transform_points, cart2spher


class SampleSIRE(Transform):
    """ Return a monai-styled sampled sphere with different scales. Used as
    base class and for inference. """

    def __init__(
        self,
        npoints: int = 32,
        alpha: float = 3,
        r: float = 0.3,
        subdivisions: int = 3,
        stratify_radius: bool = False,
        device: str = "cpu",
    ):
        super(SampleSIRE, self).__init__()

        self.npoints = npoints
        self.alpha = alpha
        self.r = r
        self.stratify_radius = stratify_radius
        self.device = device

        self.sphere = IcoSphere(subdivisions=subdivisions)
        self.sphereverts = torch.from_numpy(self.sphere.sphereverts).float()[
            :, 1:] - torch.tensor([np.pi / 2, 0])  # Only [phi, theta] keeped. range of phi: [-0.5pi, 0.5pi]

    def _get_spheres(
        self,
        image: torch.Tensor,  # Dimension order [D, H, W]
        affine: torch.Tensor,
        spheres: Data,
        # Should be WORLD coordinates! Coordinate order [W, H, D]
        center: torch.Tensor,
    ):

        # Convert center from world coordinates to image coordinates
        image_center = transform_points(
            center.view(-1, 3), torch.linalg.inv(affine))

        # Compute casting rays for spheres
        rays = spheres.coords + image_center
        # Convert sampling coordinates to fit torch.nn.functional.grid_sample()
        rays = (
            rays * (2 / torch.flip(torch.tensor(image.shape).to(self.device), dims=[0])) - 1).float()

        # Sample features from the underlying image
        spheres = copy.deepcopy(spheres)
        spheres.center = image_center
        spheres.features = (
            torch.nn.functional.grid_sample(
                # [D, H, W] dimension order, corresponding to [Z, Y, X]
                image[None, None, ...],
                # [W, H, D] coordinate order, corresponding to [X, Y, Z]
                rays.view(1, 1, 1, rays.shape[0], 3),
                padding_mode="reflection",  # TODO: How will this influence performance?
                align_corners=True,
            )
            .squeeze()
            .reshape(-1, self.npoints)
        )

        return spheres

    def __call__(self, data: Dict[str, Any], point: torch.tensor, labels: Tuple[str] = ("label",)):

        data = move_data_to_device(data, self.device)
        point = move_data_to_device(point, self.device)

        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        scales = data["graph_meta_dict"]["scales"]
        nverts = data["graph_meta_dict"]["nverts"]
        spheres = copy.deepcopy(data["graph"])

        sampled_spheres = self._get_spheres(image, affine, spheres, point)

        return {
            "global": {
                "affine": affine,
                "scales": scales,
                "nverts": nverts,
            },
            "sample": {
                "spheres": sampled_spheres,
                "center": point,
                "index": torch.tensor([0]),
            },
        }


class SampleSIREImageOnly(SampleSIRE):
    """ Sample image only. """

    def _get_direction_heatmap(self, directions: torch.Tensor):

        directions = cart2spher(directions)[:, 1:] - \
            torch.tensor([np.pi / 2, 0])

        # Relabel closest vertex to be heatmap peak
        dists = haversine_distances(self.sphereverts, directions)
        dists = self.sphereverts[np.argmin(dists, axis=0), :]

        # Calculate heatmap
        dists = haversine_distances(self.sphereverts, dists)
        dists_discrete = torch.min(torch.from_numpy(dists).float(), dim=1)[0]
        dists_discrete = torch.clamp(
            (torch.exp(self.alpha * (1 - dists_discrete / self.r))) *
            (dists_discrete < self.r).long(),
            0,
            np.exp(self.alpha),
        )

        return dists_discrete

    def _get_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:

        index = torch.tensor(data["index"])
        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        scales = data["graph_meta_dict"]["scales"]
        nverts = data["graph_meta_dict"]["nverts"]
        center = torch.tensor(data["center"])
        augmentation = data["augmentation"]  # Whether augmentation

        # Local copies
        graph_data = copy.deepcopy(data["graph"])

        # Calculate spheres and direction heatmap
        if not augmentation:
            tangents = torch.tensor(data["tangents"])
            heatmap = self._get_direction_heatmap(tangents)
        else:
            tangents = torch.zeros((2, 3))
            heatmap = torch.zeros(len(self.sphereverts), 1)  # Zero response

        sampled_spheres = self._get_spheres(image, affine, graph_data, center)

        return {
            "global": {"affine": affine, "scales": scales, "nverts": nverts},
            "sample": {
                "index": index,
                "spheres": sampled_spheres,
                "sampled_center": center.squeeze(),
                "directions": tangents,
                "target": heatmap.squeeze(),
                "augmentation": augmentation,
            },
        }

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        return self._get_sample(data)
