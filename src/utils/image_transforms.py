# Transformation during inference
import copy
from typing import Dict, Any

import torch
from torch_geometric.data import Data
from monai.transforms import (
    Transform,
    Compose,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
)
from pytorch_lightning.utilities import move_data_to_device

from src.utils.config import BaseConfig
from src.utils.geometry import IcoSphere, transform_points


class SampleIcoSphere(Transform):
    """
    Sample image features and project to Icosphere graph nodes.
    """

    def __init__(self, config: BaseConfig):

        super().__init__()
        self.config = config
        self.subdivisions = config.subdivisions
        self.npoints = config.npoints
        self.device = config.device

        self.sphere = IcoSphere(subdivisions=self.subdivisions)

    def get_features(
        self,
        image: torch.Tensor,
        affine: torch.Tensor,
        graph: Data,
        phy_center: torch.Tensor,
    ):

        DEV = phy_center.device

        img_center = transform_points(
            phy_center.view(-1, 3), torch.linalg.inv(affine))
        shpT = torch.tensor(image.shape[::-1]).to(DEV)  # (W, H, D)
        rays: torch.Tensor = (graph.coords + img_center).to(DEV)
        rays = (rays * (2 / shpT) - 1).float()

        graph.center = img_center
        graph.features = torch.nn.functional.grid_sample(
            # [D, H, W] dimension order, corresponding to [Z, Y, X]
            image[None, None, ...],
            # [W, H, D] coordinate order, corresponding to [X, Y, Z]
            rays.view(1, 1, 1, rays.shape[0], 3),
            padding_mode="reflection",
            align_corners=True,
        ).squeeze().view(-1, self.npoints)

        return graph

    def __call__(self, data: Dict[str, Any], point: torch.tensor):

        data = move_data_to_device(data, self.device)
        point = move_data_to_device(point, self.device)

        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        scales = data["graph_meta_dict"]["scales"]
        nverts = data["graph_meta_dict"]["nverts"]
        graph = copy.deepcopy(data["graph"])

        sampled_graph = self.get_features(image, affine, graph, point)

        return {
            "global": {
                "affine": affine,
                "scales": scales,
                "nverts": nverts,
            },
            "sample": {
                "graph": sampled_graph,
                "center": point,
                "index": torch.tensor([0]),
            },
        }


class SampleIcoSphereMultiDir(SampleIcoSphere):
    """
    Sample image and project features to graph nodes. Generate multi-direction
    labels as node indices.
    """

    def _get_direction_label(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Assign classification label to sphere.
        """

        directions = directions.view(-1, 3)  # (2, 3) or (N, 3)
        verts = torch.from_numpy(self.sphere.cartverts).view(-1, 3)

        cls_label = torch.zeros((len(self.sphere.cartverts), ))
        for direction in directions:
            dists = torch.linalg.norm(verts - direction.view(-1, 3), dim=-1)
            cls_label[torch.argmin(dists.squeeze())] = 1.0

        return cls_label

    def _get_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:

        index = torch.tensor(data["index"])
        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        center = torch.tensor(data["center"])
        augmentation = data["augmentation"]  # Whether augmentation
        base_radius = data["base_radius"]  # noqa

        # Local copies
        graph = copy.deepcopy(data["graph"])

        # Calculate spheres and direction heatmap
        if not augmentation:
            tangents = torch.tensor(data["tangents"])
            target = self._get_direction_label(tangents)
            radius = torch.tensor(data["radius"])
            # TEST: We transform the target to logit space to keep gradient
            radius = radius / base_radius
            radius = torch.log(radius / (1 - radius))
        else:
            radius = torch.tensor(-1.0)
            target = torch.zeros((len(self.sphere.cartverts), ))

        sampled_graph = self.get_features(image, affine, graph, center)

        return sampled_graph, target, torch.tensor(augmentation), index, radius

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        return self._get_sample(data)


class InferImgPreTransform(Transform):
    """ A wrapper of IcoSpherical data transformation pre-inference. """

    def __init__(self, config):

        super().__init__()
        self.config = config

        if self.config.WINDOW_LEVEL is None or self.config.WINDOW_WIDTH is None:
            self.pre_transform = Compose(
                [
                    ScaleIntensityRangePercentilesd(
                        keys=["image"],
                        lower=0.5,
                        upper=99.5,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                ],
            )
        else:
            self.pre_transform = Compose(
                [
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=self.config.WINDOW_LEVEL -
                        (self.config.WINDOW_WIDTH / 2),
                        a_max=self.config.WINDOW_LEVEL +
                        (self.config.WINDOW_WIDTH / 2),
                        b_min=0.0,
                        b_max=1.0,
                        clip=True
                    ),
                ]
            )

    def __call__(self, data: Dict[str, Any], **kwds) -> Dict[str, Any]:

        data = self.pre_transform(data)

        return data
