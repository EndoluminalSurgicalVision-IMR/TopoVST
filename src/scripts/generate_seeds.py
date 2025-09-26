from typing import Dict, Any

import torch
import numpy as np
from scipy import ndimage
from skimage.measure import label
import SimpleITK as sitk

from src.utils.geometry import get_affine, transform_points


class SegmentorProcessor:
    """Base class handling image segmentations and seed points generation."""

    def __init__(self, device: str = "cpu") -> None:

        self.device = device

    def prepare_mask_affine(self, image_path: str, mask_path: str):
        """ Prepare mask, affine for segmentor processor """

        itk_image = sitk.ReadImage(image_path)
        affine, _ = get_affine(itk_image)
        affine = affine.to(self.device)
        if mask_path is None:
            mask = None  # Get segmentation mask via segmentation models
        # Direct mask images are used if available
        else:
            itk_mask = sitk.ReadImage(mask_path)
            mask = sitk.GetArrayFromImage(itk_mask)

        return mask, affine

    def get_seeds(self, mask: np.ndarray, affine: torch.Tensor):
        """Seed points extraction from segmentation masks.

        Args:
            mask (np.ndarray): segmentation mask with binary labels.
            affine (torch.Tensor): affine matrix for current mask image.
        """

        raise NotImplementedError("Seed generation method undefined.")

    def __call__(self, *args, **kwds):

        raise NotImplementedError(
            "Calling behavior undefined for class ." "%s" % __class__.__name__
        )


class MaxDistTransSegmentorProcessor(SegmentorProcessor):
    """ Segmentor processor that finds maximum distance transform point on
     each connected regions. """

    NUM_CONN_COMPONENT = None

    def get_seeds(self, mask: np.ndarray, affine: torch.Tensor) -> np.ndarray:

        # Input mask should be in [W, H, D] spatial order.
        # Extract NUM_CONN_COMPONENT largest connected regions
        conn_map = label(mask, connectivity=3)
        count = np.bincount(conn_map.flatten())[1:]  # Remove 0 background
        # If NUM_CONN_COMPONENT is None, extract from all connected regions
        if self.NUM_CONN_COMPONENT is None:
            conn_labels = np.argsort(count).tolist()
        else:
            conn_labels = np.argsort(
                count)[-1 * self.NUM_CONN_COMPONENT:].tolist()
        conn_labels = [label_num + 1 for label_num in conn_labels]
        # Get the position of maximum distance transform value.
        sampling = (abs(affine[0, 0].item()), abs(
            affine[1, 1].item()), abs(affine[2, 2].item()))

        seeds_list = []
        for conn_label in conn_labels:
            conn_region = np.where(conn_map == conn_label, 1, 0)
            conn_region_dt = ndimage.distance_transform_edt(
                conn_region, sampling)
            max_dt_value = np.amax(conn_region_dt)
            seeds_list.append(
                np.argwhere(conn_region_dt == max_dt_value).reshape(-1, 3)[0])

        print("Number of seeds: ", len(seeds_list))
        seeds = np.array(seeds_list).reshape(-1, 3)
        return transform_points(
            torch.from_numpy(seeds).view(-1, 3).to(self.device),
            affine).cpu().numpy()

    def __call__(self, image_path: str, mask_path: str = None) -> Dict[str, Any]:

        mask, affine = self.prepare_mask_affine(image_path, mask_path)
        seeds = self.get_seeds(mask.T, affine)

        return {
            "seeds": seeds.tolist(),
            "affine": affine.cpu().tolist(),
        }


if __name__ == "__main__":

    import os
    import json
    from typing import List

    # NOTE: The code below is only for reference
    IMAGE = ""
    PRED = ""
    SPLITS = ""  # Train/Test/Val Splits
    DST = ""
    NUM_CONN_COMPONENT = 1
    DEVICE = "cuda:0"

    def get_imgPath(case_name: str) -> str:

        return os.path.join("imagesTr", "%s_0000.nii.gz" % case_name)

    def get_maskPath(case_name: str) -> str:

        return os.path.join("labelsTr", "%s_0000.nii.gz" % case_name)

    def get_predPath(case_name: str) -> str:

        return "%s.nii.gz" % case_name

    with open(SPLITS, 'r') as f:
        cases: List[str] = json.load(f)["test"]
        cases = [name.removesuffix("_0000.nii.gz") for name in cases]
    cases.sort()
    images = [os.path.join(IMAGE, get_imgPath(case)) for case in cases]
    masks = [os.path.join(PRED, get_predPath(case)) for case in cases]

    dt_preprocessor = MaxDistTransSegmentorProcessor(device=DEVICE)
    dt_preprocessor.NUM_CONN_COMPONENT = NUM_CONN_COMPONENT
    dt_dir_name = f"maxDT_from_pred_{NUM_CONN_COMPONENT}_largest"

    preprocessor = dt_preprocessor
    dir_name = dt_dir_name
    for case_id, mask, image in zip(cases, masks, images):
        print("Processing %s..." % case_id)
        res = preprocessor(image, mask)
        num_seeds = len(res["seeds"])

        directory = os.path.join(DST, dir_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, f"{case_id}_seeds_{num_seeds}.json"), 'w') as f:
            json.dump(res, f)
