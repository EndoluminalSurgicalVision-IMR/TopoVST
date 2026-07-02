import copy
from typing import Dict, Tuple, Any, Callable, List

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch.nn.functional import grid_sample
from monai.transforms import Randomizable, Transform
from pytorch_lightning.utilities import move_data_to_device

from src.utils.geometry import transform_points, IHPSphere, get_rotation_matrix


class SampleIhpSphere(Transform, Randomizable):
    """ Return a monai-styled sampled sphere with specific scale. Used as base
    class and for inference. """

    def __init__(
        self,
        npoints: int = 64,
        level: int = 4,
        device: str = "cpu",
    ):

        self.npoints = npoints
        self.device = device
        self.sphere = IHPSphere(subdivision=level)

    def _get_sphere(
        self,
        image: torch.Tensor,  # Dimension order [D, H, W]
        affine: torch.Tensor,
        sphere: Data,
        center: torch.Tensor,
        rotMatrix: torch.Tensor = torch.eye(4, dtype=torch.float32)
    ) -> Data:

        shpT = torch.tensor(image.shape[::-1]).to(rotMatrix.device)

        phy_coords = sphere.coords.view(-1, 3) + center.view(-1, 3)
        phy_coords = transform_points(phy_coords, rotMatrix)  # Rand rotation
        img_coords = transform_points(phy_coords, torch.linalg.inv(affine))
        s_coords = (img_coords * (2 / shpT) - 1).float()

        # Sample features from the underlying image
        sphere.features = grid_sample(
            image[None, None, ...],
            s_coords.view(1, 1, 1, s_coords.shape[0], 3),
            padding_mode="reflection",
            align_corners=True,
        ).squeeze().reshape(-1, self.npoints)

        return sphere

    def __call__(self, data: Dict[str, Any], point: torch.Tensor):

        data = move_data_to_device(data, self.device)
        point = move_data_to_device(point, self.device)

        data_id = data["id"]
        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        nverts = data["sphere_meta_dict"]["nverts"]
        sphere = copy.deepcopy(data["sample_sphere"])
        rot = torch.eye(4, dtype=torch.float32, device=self.device)

        sampled_sphere = self._get_sphere(image, affine, sphere, point, rot)

        return {
            "global": {
                "id": data_id,
                "image": image,
                "affine": affine,
                "nverts": nverts,
            },
            "sample": {
                "sphere": sampled_sphere,
                "center": point,
                "index": torch.tensor([0])
            }
        }


class SampleCOACTTracker(SampleIhpSphere):
    """
    Sampler for COACT Tracker Sphere.
    """

    def _get_direction_label(
        self,
        directions: torch.Tensor,
        rotMatrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Assign classification label to sphere vertices.
        """

        directions = directions.view(-1, 3)  # (2, 3) or (N, 3)
        verts = torch.from_numpy(self.sphere.cartverts).view(-1, 3)
        verts = transform_points(verts, rotMatrix)  # Rotate the sphere

        cls_label = torch.zeros((len(verts), ))
        for direction in directions:
            dists = torch.linalg.norm(verts - direction.view(-1, 3), dim=-1)
            cls_label[torch.argmin(dists.squeeze())] = 1.0

        return cls_label

    def _get_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:

        index = torch.tensor(data["index"])
        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        center = torch.tensor(data["center"])
        rotate = data["rotate"]  # Whether to use rotation augmentation
        directions = torch.tensor(data["tangents"]).view(-1, 3)

        if not rotate:
            rotMat = torch.eye(4, dtype=torch.float32)
        else:
            norm_vec = np.random.random(size=(1, 3))
            norm_vec /= np.linalg.norm(norm_vec, ord=2)
            rotMat = get_rotation_matrix(
                torch.from_numpy(norm_vec).squeeze())
            rotMat = torch.concat((rotMat, torch.zeros((3, 1))), dim=-1)
            rotMat = torch.concat(
                (rotMat, torch.tensor([[0.0, 0.0, 0.0, 1.0]])), dim=0)
        sphere = copy.deepcopy(data["sample_sphere"])
        sample = self._get_sphere(image, affine, sphere, center, rotMat)
        label = self._get_direction_label(directions, rotMat)

        return index, sample, label.squeeze()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        return self._get_sample(data)
