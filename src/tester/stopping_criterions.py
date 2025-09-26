import os
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import edt
import SimpleITK as sitk

from src.utils.geometry import transform_points


class StoppingCriterionBase:

    def __call__(self, **kwargs) -> bool:
        """
        Evaluates current stopping criterion.
        """

        raise NotImplementedError()

    def update(self, **kwds) -> None:
        """
        Update attributes of current stopping criterion.
        """

        pass

    def __str__(self):
        """
        Return str representation of current stopping criterion.
        """

        return f"{self.__class__.__name__}()"


class MaxIterationsStoppingCriterion(StoppingCriterionBase):

    def __init__(self, max_iterations: int):

        self.max_iterations = max_iterations

    def __call__(self, **kwds):

        iteration = kwds.get("iteration", None)
        if iteration is None:
            return False

        return self.max_iterations <= iteration

    def __str__(self):

        return f"{self.__class__.__name__}(max_iterations={self.max_iterations})"


class MaxProbStoppingCriterion(StoppingCriterionBase):

    def __init__(self, min_prob: float):

        self.min_prob = min_prob

    def __call__(self, **kwds):

        pred = kwds.get("probabilities", None)
        if pred is None:
            return False

        return np.amax(pred) <= self.min_prob

    def __str__(self):

        return f"{self.__class__.__name__}(min_probs={self.min_prob})"


class InvalidRegionStoppingCriterion(StoppingCriterionBase):

    def __init__(self):

        self.invalid_mask = None

    def update(self, shape: Tuple, arr: np.ndarray | torch.Tensor, affine: torch.Tensor):

        self.invalid_mask = arr if isinstance(
            arr, torch.Tensor) else torch.from_numpy(arr)  # (D, H, W)
        self.affine = affine
        self.shape = torch.tensor(shape)  # (W, H, D)

    def __call__(self, point: torch.Tensor, **kwargs):

        if self.invalid_mask is None:
            return False

        coords = transform_points(
            point.view(-1, 3), torch.linalg.inv(self.affine.to(point.device)))
        resized_coord = 2 * coords / self.shape.to(point.device) - 1
        resized_coord = resized_coord.view(-1, 3)
        res = torch.nn.functional.grid_sample(
            self.invalid_mask[None, None, ...].to(point.device).float(),
            resized_coord.view(1, 1, 1, len(resized_coord), 3),
            "nearest",
            padding_mode="reflection",
            align_corners=True,
        ).squeeze().item()

        return res > 0

    def __str__(self):

        return f"{self.__class__.__name__}"


class MaxEntropyStoppingCriterion(StoppingCriterionBase):

    def __init__(self, max_entropy: float, epsilon: float = 0.001) -> None:

        self.max_entropy = max_entropy
        self.epsilon = epsilon

        self.moving_avg_ent = deque([], maxlen=5)  # Moving Average is used

    def update(self):
        """ Clear the Queue of entropies. """

        self.moving_avg_ent.clear()

    def __call__(self, heatmap: torch.Tensor, **kwargs) -> bool:

        logits = heatmap.squeeze()
        probs = torch.nn.functional.softmax(logits, dim=0)
        # Calculate normalized entropy value
        max_ent = torch.log2(torch.tensor(len(probs)))
        entropy = torch.sum(-1 * probs * torch.log2(probs)) / max_ent

        if len(self.moving_avg_ent) == 0:
            for _ in range(5):
                self.moving_avg_ent.append(entropy.item())
        else:
            self.moving_avg_ent.append(entropy.item())

        return np.average(self.moving_avg_ent) >= self.max_entropy

    def __str__(self):

        return f"{self.__class__.__name__}(current_entropy={np.average(self.moving_avg_ent)}, max_entropy={self.max_entropy})"


class EndPointsStoppingCriterion(StoppingCriterionBase):

    def __init__(self, points: np.array, distance: float = 1):

        self.points = torch.from_numpy(points) if points is not None else None
        self.distance = distance

    def update(self, new_points: torch.Tensor | None):

        if self.points is None or new_points is None:  # At initialization
            self.points = new_points
        elif new_points is not None:
            # self.points = torch.cat([self.points, new_points])
            self.points = new_points.clone()

    def __call__(self, **kwds):

        if self.points is None:
            return False
        point = kwds.get("point", None)
        if point is None:
            return False
        point = point.to(self.points.device)
        distances = torch.linalg.norm(self.points - point.reshape(3), dim=1)
        return torch.any(distances < self.distance).item()

    def __str__(self):

        return f"{self.__class__.__name__}(distance={self.distance})"


class AlreadyTrackedStoppingCriterion(EndPointsStoppingCriterion):

    pass


class NoValidDirectionsStoppingCriterion(StoppingCriterionBase):
    """ Stopping criterion based on empty valid directions after direction
     filtering. """

    def __init__(self):

        pass

    def __call__(self, directions: List, **kwargs):

        return len(directions) == 0

    def __str__(self):

        return f"{self.__class__.__name__}"


class OutofSegmentStoppingCriterion(StoppingCriterionBase):
    """
    Stopping criterion based on tracker position. Stop if tracker steps out of
    segmentation result.
    """

    def __init__(self, min_sd: float = -1.0):

        self.sdf = None
        self.affine = None
        self.threshold = min_sd  # Unit:mm

    def update(
        self,
        segmentation: str | np.ndarray,
        spacing: torch.Tensor,
        affine: torch.Tensor,
    ):
        """
        Construct signed-distance field from the segmentation. Input segmenta-
        tion image should be of shape (D, H, W). Spacing should be in order of
        (W, H, D).
        """

        if isinstance(segmentation, str):
            itk_mask = sitk.ReadImage(segmentation, sitk.sitkUInt8)
            mask = sitk.GetArrayFromImage(itk_mask)
        elif isinstance(segmentation, np.ndarray):
            mask = segmentation.astype(np.uint8)
        else:
            raise TypeError("Input segmentation must be either str or array.")
        self.shape = torch.tensor(mask.shape[::-1])  # (W, H, D)
        self.affine = affine.clone()
        self.sdf = torch.from_numpy(edt.sdf(
            mask,
            anisotropy=spacing.squeeze().tolist()[::-1],
            black_border=True,
            parallel=os.cpu_count() // 4,
        )).float()

    def __call__(self, **kwds):

        point = kwds.get("point", None)
        if self.sdf is None:
            return False
        if point is None:
            return False

        self.affine = self.affine.to(point.device)
        self.shape = self.shape.to(point.device)
        self.sdf = self.sdf.to(point.device)
        img_coord = transform_points(
            point.view(-1, 3), torch.linalg.inv(self.affine))
        img_coord = 2 * img_coord / self.shape - 1
        signed_dist = torch.nn.functional.grid_sample(
            self.sdf[None, None, ...],
            img_coord.view(1, 1, 1, len(img_coord), 3),
            "bilinear",
            padding_mode="reflection",
            align_corners=True,
        ).squeeze().item()

        return signed_dist <= self.threshold

    def __str__(self):

        return f"{self.__class__.__name__}(min_sd={self.threshold})"


class OutOfBoundaryStoppingCriterion(StoppingCriterionBase):
    """ Stopping criterion based on tracker position inside image boundaries
    or not. """

    def __init__(self):

        pass

    def update(self, shape: Tuple, affine: torch.Tensor):

        self.shape = torch.tensor(shape)
        self.affine = affine

    def __call__(self, **kwds):

        point = kwds.get("point", None)
        if point is None:
            return False

        img_coord = transform_points(
            point.view(-1, 3), torch.linalg.inv(self.affine.to(point.device)))

        return torch.any(img_coord >= self.shape.to(point.device)).item() or torch.any(img_coord <= 0).item()

    def __str__(self):

        return f"{self.__class__.__name__}(shape={self.shape})"
