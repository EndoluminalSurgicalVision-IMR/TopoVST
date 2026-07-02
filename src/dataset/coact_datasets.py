import os
import ast
import random
from typing import List, Dict, Any

import pandas as pd
from torch_geometric.data import Dataset
from monai.transforms import ScaleIntensityRanged

from src.utils.load_transforms import LoadImageMask
from src.dataset.coact_preprocessor import BuildCOACTSphere
from src.dataset.coact_sampler import SampleCOACTTracker


class COACTCoWData(Dataset):
    """
    Dataset for COACT on CoW (Circle of Willis, Dataset033_BinTopCoW24_CT).

    Folder layout (nnUNet-style, already binarized):
        images{Tr|Ts}/{case}_0000.nii.gz, labels{Tr|Ts}/{case}.nii.gz
    """

    WINDOW_LEVEL = 200
    WINDOW_WIDTH = 600
    TRN_LEN = 20000
    VAL_LEN = 500

    def __init__(self, config, stage: str):

        super().__init__()
        self.stage = stage
        self.src_dir = config.src_dir
        # Load samples
        self.sample_files: Dict[str, str] = config.samples[stage]
        self.load_samples(augmentation=config.augmentation, stage=stage)

        # Set training transforms
        self._set_transform(
            ray_len=config.ray_len,
            npoints=config.npoints,
            level=config.level,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:

        sample = self.samples[index]
        return self.prepare_data(sample, index)

    def __len__(self):

        return len(self.samples)

    def load_samples(self, **kwds):
        """
        Load from files the training samples to use. Augmentation only contains
        """

        LOAD_LEN = self.TRN_LEN if kwds["stage"] == "train" else self.VAL_LEN

        use_aug = kwds.get("augmentation", True)  # Use augmentation or not
        no_aug_df = pd.read_csv(self.sample_files["on_centerline"], sep=",")
        no_aug: List[Dict] = no_aug_df.to_dict(orient="records")

        if not use_aug or self.stage in ["val", "test"]:
            random.shuffle(no_aug)
            self.samples = no_aug[:LOAD_LEN]
            return

        off_cl_df = pd.read_csv(self.sample_files["off_centerline"], sep=",")
        off_cl: List[Dict] = off_cl_df.to_dict(orient="records")
        random.shuffle(off_cl)
        self.samples = no_aug[:int(0.9 * LOAD_LEN)] + \
            off_cl[:int(0.1 * LOAD_LEN)]
        random.shuffle(self.samples)

    def prepare_data(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Transform sample record to tensor for training.
        """

        case_name: str = sample["case"]
        aug: bool = sample["radius"] <= 0  # <= 0 radius means aug

        rotate = False
        if random.random() <= 0.5:
            rotate = True

        # CoW Dataset033 layout: imagesTr/Ts + labelsTr/Ts (already binarized).
        suffix = "Tr" if self.stage in ["val", "train"] else "Ts"

        data = {
            "index": index,
            "dataset": sample["dataset"],
            "case_name": case_name,
            "image": os.path.join(self.src_dir, f"images{suffix}",
                                  f"{case_name}_0000.nii.gz"),
            "mask": os.path.join(self.src_dir, f"labels{suffix}",
                                 f"{case_name}.nii.gz"),
            "center": ast.literal_eval(sample["position"]),
            "radius": sample["radius"] if not aug else None,
            "rotate": rotate,
            "augmentation": aug,
            "tangents": ast.literal_eval(sample["tangents"]) if not aug else None,
        }
        data = self.load_transform(data)
        data = self.intensity_transform(data)
        data = self.graph_transform(data)
        data = self.sampler(data)

        return data

    def _set_transform(self, **kwds):

        self.load_transform = LoadImageMask()
        self.intensity_transform = ScaleIntensityRanged(
            keys=["image"],
            a_min=self.WINDOW_LEVEL - (self.WINDOW_WIDTH / 2),
            a_max=self.WINDOW_LEVEL + (self.WINDOW_WIDTH / 2),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )

        # Prepare GNN data structure
        self.graph_transform = BuildCOACTSphere(
            ray_len=kwds["ray_len"],
            npoints=kwds["npoints"],
            level=kwds["level"],
        )

        # Fill in input image sample
        self.sampler = SampleCOACTTracker(
            npoints=kwds["npoints"],
            level=kwds["level"],
        )
