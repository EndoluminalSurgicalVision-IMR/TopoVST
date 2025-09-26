import os
import ast
import random
from typing import Dict, List, Any

import pandas as pd
from torch_geometric.data import Dataset
from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
)

from src.utils.load_transforms import LoadImageMask, LoadImageMaskRTCached
from src.utils.graph_transforms import (
    BuildRandScales,
    BuildMultiScaleGraphStructure,
    FillSamplingCoords,
    BuildForGEMGCN,
)
from src.utils.image_transforms import SampleIcoSphereMultiDir


class AdaSIREATM22Data(Dataset):
    """
    Ada-SIRE Dataset for ATM22.
    """

    def __init__(self, config, stage: str):

        super().__init__()
        self.window_level = config.window_level
        self.window_width = config.window_width
        self.trn_len = config.train_samples
        self.val_len = config.val_samples
        self.stage = stage
        self.src_dir = config.src_dir
        # Load samples
        self.sample_files: Dict[str, str] = config.samples[stage]
        self.load_samples(augmentation=config.augmentation, stage=stage)

        self.config = config
        # Set training transforms
        self._set_transform()

    def __getitem__(self, index: int) -> Dict[str, Any]:

        sample = self.samples[index]
        return self.prepare_data(sample, index)

    def __len__(self):

        return len(self.samples)

    def load_samples(self, **kwds):
        """
        Load from files the training samples to use.
        """

        print("Loading samples ...")
        LOAD_LEN = self.trn_len if kwds["stage"] == "train" else self.val_len

        use_aug = kwds.get("augmentation", True)  # Use augmentation or not
        no_aug_df = pd.read_csv(self.sample_files["on_centerline"], sep=",")
        no_aug: List[Dict] = no_aug_df.to_dict(orient="records")

        if not use_aug or self.stage in ["val", "test"]:
            random.shuffle(no_aug)
            self.samples = no_aug[:LOAD_LEN]
            return

        off_cl_df = pd.read_csv(self.sample_files["off_centerline"], sep=",")
        off_cl: List[Dict] = off_cl_df.to_dict(orient="records")
        out_lumen_df = pd.read_csv(self.sample_files["out_lumen"], sep=",")
        out_lumen: List[Dict] = out_lumen_df.to_dict(orient="records")
        random.shuffle(off_cl)
        random.shuffle(out_lumen)
        self.samples = no_aug[:int(0.6 * LOAD_LEN)] + \
            off_cl[:int(0.3 * LOAD_LEN)] + out_lumen[:int(0.1 * LOAD_LEN)]
        random.shuffle(self.samples)

        print("Loading complete.")

    def prepare_data(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Transform sample record to tensor for training.
        """

        case_name: str = sample["case"]
        aug: bool = sample["radius"] <= 0  # <= 0 radius means aug

        data = {
            "index": index,
            "dataset": sample["dataset"],
            "case_name": case_name,
            "image": os.path.join(self.src_dir, "imagesTr",
                                  f"{case_name}_0000.nii.gz"),
            "mask": os.path.join(self.src_dir, "labelsTr",
                                 f"{case_name}_0000.nii.gz"),
            "center": ast.literal_eval(sample["position"]),
            "radius": sample["radius"] if not aug else None,
            "base_radius": self.config.base_radius,
            "augmentation": aug,
            "tangents": ast.literal_eval(sample["tangents"]) if not aug else None,
        }
        data = self.load_transform(data)
        data = self.pre_transform(data)

        return data

    def _set_transform(self):

        self.load_transform = LoadImageMask()
        self.pre_transform = Compose([
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.window_level - (self.window_width / 2),
                a_max=self.window_level + (self.window_width / 2),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            BuildRandScales(self.config),
            BuildMultiScaleGraphStructure(self.config),
            FillSamplingCoords(self.config),
            BuildForGEMGCN(keys=["graph"]),
            SampleIcoSphereMultiDir(self.config),
        ])


class AdaSIREAorta24Data(Dataset):
    """
    Ada-SIRE Dataset for Aorta24
    """

    def __init__(self, config, stage: str):

        super().__init__()
        self.window_level = config.window_level
        self.window_width = config.window_width
        self.trn_len = config.train_samples
        self.val_len = config.val_samples
        self.stage = stage
        self.src_dir = config.src_dir
        # Load samples
        self.sample_files: Dict[str, str] = config.samples[stage]
        self.load_samples(augmentation=config.augmentation, stage=stage)

        self.config = config
        # Set training transforms
        self._set_transform()

    def __getitem__(self, index: int) -> Dict[str, Any]:

        sample = self.samples[index]
        return self.prepare_data(sample, index)

    def __len__(self):

        return len(self.samples)

    def load_samples(self, **kwds):
        """
        Load from files the training samples to use.
        """

        print("Loading samples ...")
        LOAD_LEN = self.trn_len if kwds["stage"] == "train" else self.val_len

        use_aug = kwds.get("augmentation", True)  # Use augmentation or not
        no_aug_df = pd.read_csv(self.sample_files["on_centerline"], sep=",")
        no_aug: List[Dict] = no_aug_df.to_dict(orient="records")

        if not use_aug or self.stage in ["val", "test"]:
            random.shuffle(no_aug)
            self.samples = no_aug[:LOAD_LEN]
            return

        off_cl_df = pd.read_csv(self.sample_files["off_centerline"], sep=",")
        off_cl: List[Dict] = off_cl_df.to_dict(orient="records")
        out_lumen_df = pd.read_csv(self.sample_files["out_lumen"], sep=",")
        out_lumen: List[Dict] = out_lumen_df.to_dict(orient="records")
        random.shuffle(off_cl)
        random.shuffle(out_lumen)
        self.samples = no_aug[:int(0.6 * LOAD_LEN)] + \
            off_cl[:int(0.3 * LOAD_LEN)] + out_lumen[:int(0.1 * LOAD_LEN)]
        random.shuffle(self.samples)

        print("Loading complete.")

    def prepare_data(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Transform sample record to tensor for training.
        """

        case_name: str = sample["case"]
        aug: bool = sample["radius"] <= 0  # <= 0 radius means aug

        if self.stage in ["val", "train"]:
            suffix = "Tr"
        else:
            suffix = "Ts"

        data = {
            "index": index,
            "dataset": sample["dataset"],
            "case_name": case_name,
            "image": os.path.join(self.src_dir, f"images{suffix}",
                                  f"{case_name}_0000.nii.gz"),
            "mask": os.path.join(self.src_dir, f"labels{suffix}_bin",
                                 f"{case_name}.nii.gz"),
            "center": ast.literal_eval(sample["position"]),
            "radius": sample["radius"] if not aug else None,
            "base_radius": self.config.base_radius,
            "augmentation": aug,
            "tangents": ast.literal_eval(sample["tangents"]) if not aug else None,
        }
        data = self.load_transform(data)
        data = self.pre_transform(data)

        return data

    def _set_transform(self):

        self.load_transform = LoadImageMask()
        self.pre_transform = Compose([
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.window_level - (self.window_width / 2),
                a_max=self.window_level + (self.window_width / 2),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            BuildRandScales(self.config),
            BuildMultiScaleGraphStructure(self.config),
            FillSamplingCoords(self.config),
            BuildForGEMGCN(keys=["graph"]),
            SampleIcoSphereMultiDir(self.config),
        ])


class AdaSIREASOCAData(Dataset):
    """
    Ada-SIRE Dataset for ASOCA.
    """

    def __init__(self, config, stage: str):

        super().__init__()
        self.window_level = config.window_level
        self.window_width = config.window_width
        self.trn_len = config.train_samples
        self.val_len = config.val_samples
        self.stage = stage
        self.src_dir = config.src_dir
        # Load samples
        self.sample_files: Dict[str, str] = config.samples[stage]
        self.load_samples(augmentation=config.augmentation, stage=stage)

        self.config = config
        # Set training transforms
        self._set_transform()

    def __getitem__(self, index: int) -> Dict[str, Any]:

        sample = self.samples[index]
        return self.prepare_data(sample, index)

    def __len__(self):

        return len(self.samples)

    def load_samples(self, **kwds):
        """
        Load from files the training samples to use. Augmentation only contains
        """

        LOAD_LEN = self.trn_len if kwds["stage"] == "train" else self.val_len

        use_aug = kwds.get("augmentation", True)  # Use augmentation or not
        no_aug_df = pd.read_csv(self.sample_files["on_centerline"], sep=",")
        no_aug: List[Dict] = no_aug_df.to_dict(orient="records")

        if not use_aug or self.stage in ["val", "test"]:
            random.shuffle(no_aug)
            self.samples = no_aug[:LOAD_LEN]
            return

        off_cl_df = pd.read_csv(self.sample_files["off_centerline"], sep=",")
        off_cl: List[Dict] = off_cl_df.to_dict(orient="records")
        out_lumen_df = pd.read_csv(self.sample_files["out_lumen"], sep=",")
        out_lumen: List[Dict] = out_lumen_df.to_dict(orient="records")
        random.shuffle(off_cl)
        random.shuffle(out_lumen)
        self.samples = no_aug[:int(0.6 * LOAD_LEN)] + \
            off_cl[:int(0.3 * LOAD_LEN)] + out_lumen[:int(0.1 * LOAD_LEN)]
        random.shuffle(self.samples)

    def prepare_data(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Transform sample record to tensor for training.
        """

        case_name: str = sample["case"]
        case_type, _ = case_name.split("_")
        aug: bool = sample["radius"] <= 0  # <= 0 radius means aug

        data = {
            "index": index,
            "dataset": sample["dataset"],
            "case_name": case_name,
            "image": os.path.join(self.src_dir, case_type, "CTCA",
                                  "%s.nrrd" % case_name),
            "mask": os.path.join(self.src_dir, case_type, "Annotations",
                                 "%s.nrrd" % case_name),
            "center": ast.literal_eval(sample["position"]),
            "radius": sample["radius"] if not aug else None,
            "base_radius": self.config.base_radius,
            "augmentation": aug,
            "tangents": ast.literal_eval(sample["tangents"]) if not aug else None,
        }
        data = self.load_transform(data)
        data = self.pre_transform(data)

        return data

    def _set_transform(self):

        self.load_transform = LoadImageMask()
        self.pre_transform = Compose([
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.window_level - (self.window_width / 2),
                a_max=self.window_level + (self.window_width / 2),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            BuildRandScales(self.config),
            BuildMultiScaleGraphStructure(self.config),
            FillSamplingCoords(self.config),
            BuildForGEMGCN(keys=["graph"]),
            SampleIcoSphereMultiDir(self.config),
        ])


class AdaSIREASOCAPreTrainData(Dataset):
    """
    AdaSIRE ASOCA dataset for pretraining encoder.
    """

    def __init__(self, config, stage: str):

        super().__init__()
        self.window_level = config.window_level
        self.window_width = config.window_width
        self.trn_len = config.train_samples
        self.val_len = config.val_samples
        self.stage = stage
        self.src_dir = config.src_dir
        # Load samples
        self.sample_files: Dict[str, str] = config.samples[stage]
        self.load_samples(augmentation=config.augmentation, stage=stage)

        self.config = config
        # Set training transforms
        self._set_transform()

    def __getitem__(self, index: int) -> Dict[str, Any]:

        sample = self.samples[index]
        return self.prepare_data(sample, index)

    def __len__(self):

        return len(self.samples)

    def load_samples(self, **kwds):
        """
        Load from files the training samples to use. Augmentation only contains
        """

        LOAD_LEN = self.trn_len if kwds["stage"] == "train" else self.val_len

        use_aug = kwds.get("augmentation", True)  # Use augmentation or not
        no_aug_df = pd.read_csv(self.sample_files["on_centerline"], sep=",")
        no_aug: List[Dict] = no_aug_df.to_dict(orient="records")

        if not use_aug or self.stage in ["val", "test"]:
            random.shuffle(no_aug)
            self.samples = no_aug[:LOAD_LEN]
            return

        off_cl_df = pd.read_csv(self.sample_files["off_centerline"], sep=",")
        off_cl: List[Dict] = off_cl_df.to_dict(orient="records")
        out_lumen_df = pd.read_csv(self.sample_files["out_lumen"], sep=",")
        out_lumen: List[Dict] = out_lumen_df.to_dict(orient="records")
        random.shuffle(off_cl)
        random.shuffle(out_lumen)
        self.samples = no_aug[:int(0.8 * LOAD_LEN)] + \
            off_cl[:int(0.1 * LOAD_LEN)] + out_lumen[:int(0.1 * LOAD_LEN)]
        random.shuffle(self.samples)

    def prepare_data(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Transform sample record to tensor for training.
        """

        case_name: str = sample["case"]
        case_type, _ = case_name.split("_")
        aug: bool = sample["radius"] <= 0  # <= 0 radius means aug

        data = {
            "index": index,
            "dataset": sample["dataset"],
            "case_name": case_name,
            "image": os.path.join(self.src_dir, case_type, "CTCA",
                                  "%s.nrrd" % case_name),
            "mask": os.path.join(self.src_dir, case_type, "Annotations",
                                 "%s.nrrd" % case_name),
            "center": ast.literal_eval(sample["position"]),
            "radius": sample["radius"] if not aug else None,
            "augmentation": aug,
            "tangents": ast.literal_eval(sample["tangents"]) if not aug else None,
        }
        data = self.pre_transform(data)

        return data

    def _set_transform(self):

        self.pre_transform = Compose([
            LoadImageMaskRTCached(),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.window_level - (self.window_width / 2),
                a_max=self.window_level + (self.window_width / 2),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            BuildRandScales(self.config),
            BuildMultiScaleGraphStructure(self.config),
            FillSamplingCoords(self.config),
            BuildForGEMGCN(keys=["graph"]),
            SampleIcoSphereMultiDir(self.config),
        ])
