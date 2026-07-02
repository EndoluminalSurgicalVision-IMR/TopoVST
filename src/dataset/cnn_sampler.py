from typing import List, Dict, Any

import torch
import numpy as np
from monai.transforms import Transform
from pytorch_lightning.utilities import move_data_to_device

from src.utils.geometry import (
    transform_points,
    RectanglePatch,
    FibonacciSphere,
    get_rotation_matrix,
)


class SampleCNN(Transform):
    """
    Return a sampled CNN patch that is augmented.
    """

    def __init__(
        self,
        patch_size: List[int],
        target_spacing: List[float],
        device: str = "cpu",
        npoints: int = 500,
    ):

        super(SampleCNN, self).__init__()

        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.device = device
        self.npoints = npoints

        self.patch = RectanglePatch(patch_size, target_spacing)
        self.label_sphere = FibonacciSphere(npoints=self.npoints)

    def _get_patch(
        self,
        image: torch.Tensor,
        affine: torch.Tensor,
        center: torch.Tensor,
        rotMatrix: torch.Tensor = torch.eye(4, dtype=torch.float32),
    ) -> torch.Tensor:

        shpT = torch.tensor(image.shape[::-1]).to(center.device)  # (W, H, D)

        # Convert center from world coordinates to image coordinates!
        rotMatrix = rotMatrix.to(center.device)
        phy_coords = self.patch.create_patch(center.view(-1, 3), rotMatrix)
        img_coords = transform_points(
            phy_coords.view(-1, 3), torch.linalg.inv(affine))
        s_coords = (img_coords * (2 / shpT) - 1).float()

        # sample patch
        sample = torch.nn.functional.grid_sample(
            # [D, H, W] dimension order, corresponding to [Z, Y, X]
            image[None, None, ...],
            # [W, H, D] coordinate order, corresponding to [X, Y, Z]
            s_coords.view(1, 1, 1, s_coords.shape[0], 3),
            padding_mode="reflection",
            align_corners=True,
        ).squeeze().reshape(self.patch_size)

        return sample

    def __call__(self, data: Dict[str, Any], point: torch.Tensor):

        data = move_data_to_device(data, self.device)
        point = move_data_to_device(point, self.device)

        image = data["image"]
        affine = data["image_meta_dict"]["affine"]

        patch = self._get_patch(image, affine, point)

        return {
            "global": {
                "affine": affine,
            },
            "sample": {
                "patch": patch.unsqueeze(0),  # Add a channel
                "center": point,
                "index": torch.tensor([0]),
            },
        }


class SampleCNNTracker(SampleCNN):
    """
    Sample a rectangular patch for CNN-Tracker and create directional label.
    """

    def _get_direction_label(
        self,
        directions: torch.Tensor,
        rotMatrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assign classification label to sphere.
        """

        directions = directions.view(-1, 3)  # (2, 3) or (N, 3)
        verts = torch.from_numpy(self.label_sphere.get_coords())  # (N, 3)
        verts = transform_points(verts, rotMatrix)  # Rotate the sphere

        cls_label = torch.zeros((len(verts), ))
        for direction in directions:
            dists = torch.linalg.norm(verts - direction.view(-1, 3), dim=-1)
            cls_label[torch.argmin(dists.squeeze())] = 1.0

        return cls_label

    def _get_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:

        aug = data["augmentation"]
        image = data["image"]
        affine = data["image_meta_dict"]["affine"]
        center = torch.tensor(data["center"])
        rotate = data["rotate"]  # Whether to use rotation augmentation
        directions = torch.tensor(data["tangents"]) if not aug else None
        radius = torch.tensor(
            data["radius"]) if not aug else torch.tensor(-1.0)

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
        patch = self._get_patch(image, affine, center, rotMat)
        if directions is None:
            label = torch.zeros((self.label_sphere.get_coords().shape[0], ))
        else:
            label = self._get_direction_label(directions, rotMat)

        return patch.unsqueeze(0), label.squeeze(), radius, torch.tensor(aug)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        return self._get_sample(data)
