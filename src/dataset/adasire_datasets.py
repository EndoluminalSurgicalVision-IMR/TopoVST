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


_DEFAULT_SAMPLE_MIX = {"on_centerline": 0.6, "off_centerline": 0.3, "out_lumen": 0.1}


def resolve_sample_mix(config, default=None):
    """Return the on/off/out sample mix for a config, validated to sum to 1.

    Falls back to ``default`` (or the module-wide 0.6/0.3/0.1) when the config
    does not set ``sample_mix``.
    """

    mix = getattr(config, "sample_mix", None) or default or _DEFAULT_SAMPLE_MIX
    assert abs(sum(mix.values()) - 1.0) < 1e-6, f"sample_mix must sum to 1.0: {mix}"
    assert mix["on_centerline"] > 0, "on_centerline ratio must be > 0"
    return mix


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
        mix = resolve_sample_mix(self.config)
        self.samples = no_aug[:int(mix["on_centerline"] * LOAD_LEN)] + \
            off_cl[:int(mix["off_centerline"] * LOAD_LEN)] + \
            out_lumen[:int(mix["out_lumen"] * LOAD_LEN)]
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
            "radius_param": getattr(self.config, "radius_param", "logit"),
            "r_ref": getattr(self.config, "r_ref", None),
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
        mix = resolve_sample_mix(self.config)
        self.samples = no_aug[:int(mix["on_centerline"] * LOAD_LEN)] + \
            off_cl[:int(mix["off_centerline"] * LOAD_LEN)] + \
            out_lumen[:int(mix["out_lumen"] * LOAD_LEN)]
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
            "radius_param": getattr(self.config, "radius_param", "logit"),
            "r_ref": getattr(self.config, "r_ref", None),
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


class AdaSIRECoWData(Dataset):
    """
    Ada-SIRE Dataset for CoW (Circle of Willis, Dataset033_BinTopCoW24_CT).
    """

    def __init__(self, config, stage: str):

        super().__init__()
        self.window_level = config.window_level
        self.window_width = config.window_width
        self.val_len = config.val_samples
        self.stage = stage
        self.src_dir = config.src_dir
        self.config = config
        # Load samples
        self.sample_files: Dict[str, str] = config.samples[stage]
        self.load_samples(augmentation=config.augmentation, stage=stage)

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

        Training: by default consume *all* on-centerline rows from the CSV and
        draw off/out rows in proportion dictated by `config.sample_mix`, so the
        epoch size is data-driven. When `config.max_train_samples` is set to a
        positive value AND the data-driven size would exceed it, the per-epoch
        total is capped at that value, with every bucket scaled down by the mix
        (so the on/off/out ratio is preserved). A cap of 0 / None means
        uncapped (the original behaviour).
        Val/test: fall back to a fixed-size random subset of size `val_len`.
        """

        print("Loading samples ...")

        use_aug = kwds.get("augmentation", True)
        no_aug_df = pd.read_csv(self.sample_files["on_centerline"], sep=",")
        no_aug: List[Dict] = no_aug_df.to_dict(orient="records")

        if not use_aug or self.stage in ["val", "test"]:
            random.shuffle(no_aug)
            self.samples = no_aug[:self.val_len]
            return

        off_cl_df = pd.read_csv(self.sample_files["off_centerline"], sep=",")
        off_cl: List[Dict] = off_cl_df.to_dict(orient="records")
        out_lumen_df = pd.read_csv(self.sample_files["out_lumen"], sep=",")
        out_lumen: List[Dict] = out_lumen_df.to_dict(orient="records")
        random.shuffle(no_aug)
        random.shuffle(off_cl)
        random.shuffle(out_lumen)

        mix = resolve_sample_mix(self.config)

        n_on = len(no_aug)
        target_total = int(round(n_on / mix["on_centerline"]))
        cap = int(getattr(self.config, "max_train_samples", 0) or 0)
        if cap > 0 and cap < target_total:
            # Cap the per-epoch size; scale every bucket down by the mix so the
            # on/off/out composition is preserved.
            target_total = cap
            n_on = min(int(round(target_total * mix["on_centerline"])), n_on)
        n_off = min(int(round(target_total * mix["off_centerline"])), len(off_cl))
        n_out = min(target_total - n_on - n_off, len(out_lumen))

        self.samples = no_aug[:n_on] + off_cl[:n_off] + out_lumen[:n_out]
        random.shuffle(self.samples)
        print(f"Loaded {len(self.samples)} samples "
              f"(on={n_on}, off={n_off}, out={n_out}); mix={mix}, "
              f"cap={cap if cap > 0 else 'none'}")

    def prepare_data(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Transform sample record to tensor for training.
        """

        case_name: str = sample["case"]
        aug: bool = sample["radius"] <= 0  # <= 0 radius means aug

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
            "base_radius": self.config.base_radius,
            "radius_param": getattr(self.config, "radius_param", "logit"),
            "r_ref": getattr(self.config, "r_ref", None),
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


class AdaSIREParseData(AdaSIRECoWData):
    """
    Ada-SIRE Dataset for Parse22.

    Parse22 uses the same nnUNet folder layout as CoW and its converted
    centerline graphs use the same stored RAS-style VTP convention. The
    Parse sample CSVs are an order of magnitude larger than CoW (~430K
    on-centerline rows alone), so this subclass caps the per-epoch sample
    count at ``config.train_samples`` rather than consuming every
    on-centerline row.
    """

    def load_samples(self, **kwds):

        print("Loading samples ...")

        use_aug = kwds.get("augmentation", True)
        no_aug_df = pd.read_csv(self.sample_files["on_centerline"], sep=",")
        no_aug: List[Dict] = no_aug_df.to_dict(orient="records")

        if not use_aug or self.stage in ["val", "test"]:
            random.shuffle(no_aug)
            self.samples = no_aug[:self.val_len]
            return

        off_cl_df = pd.read_csv(self.sample_files["off_centerline"], sep=",")
        off_cl: List[Dict] = off_cl_df.to_dict(orient="records")
        out_lumen_df = pd.read_csv(self.sample_files["out_lumen"], sep=",")
        out_lumen: List[Dict] = out_lumen_df.to_dict(orient="records")
        random.shuffle(no_aug)
        random.shuffle(off_cl)
        random.shuffle(out_lumen)

        mix = resolve_sample_mix(self.config)
        target_total = int(getattr(self.config, "train_samples", 0))
        if target_total <= 0:
            raise ValueError(
                "AdaSIREParseData requires config.train_samples > 0 as a "
                "per-epoch sample cap; Parse on-centerline rows alone exceed "
                "400K and the uncapped path would dominate epoch wall time.")
        n_on = min(int(round(target_total * mix["on_centerline"])), len(no_aug))
        n_off = min(int(round(target_total * mix["off_centerline"])),
                    len(off_cl))
        n_out = min(target_total - n_on - n_off, len(out_lumen))

        self.samples = no_aug[:n_on] + off_cl[:n_off] + out_lumen[:n_out]
        random.shuffle(self.samples)
        print(f"[Parse] Loaded {len(self.samples)} samples "
              f"(on={n_on}, off={n_off}, out={n_out}); "
              f"mix={mix}, cap={target_total}")


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
        mix = resolve_sample_mix(self.config)
        self.samples = no_aug[:int(mix["on_centerline"] * LOAD_LEN)] + \
            off_cl[:int(mix["off_centerline"] * LOAD_LEN)] + \
            out_lumen[:int(mix["out_lumen"] * LOAD_LEN)]
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
            "radius_param": getattr(self.config, "radius_param", "logit"),
            "r_ref": getattr(self.config, "r_ref", None),
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
        mix = resolve_sample_mix(
            self.config,
            default={"on_centerline": 0.8, "off_centerline": 0.1, "out_lumen": 0.1})
        self.samples = no_aug[:int(mix["on_centerline"] * LOAD_LEN)] + \
            off_cl[:int(mix["off_centerline"] * LOAD_LEN)] + \
            out_lumen[:int(mix["out_lumen"] * LOAD_LEN)]
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
