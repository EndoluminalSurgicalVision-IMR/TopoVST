import os
import ast
import random
from typing import Dict, List, Any

import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    ScaleIntensityRanged,
)

from src.utils.load_transforms import LoadImageMask
from src.utils.graph_transforms import (
    BuildRandScales,
    BuildMultiScaleGraphStructure,
    FillSamplingCoords,
    BuildForGEMGCN,
)
from src.dataset.sire_sampler import SampleSIREImageOnly


class SIRECoWData(Dataset):
    """
    SIRE Dataset for CoW (Circle of Willis, Dataset033_BinTopCoW24_CT).

    Folder layout (nnUNet-style, already binarized):
        images{Tr|Ts}/{case}_0000.nii.gz, labels{Tr|Ts}/{case}.nii.gz
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

    @staticmethod
    def _is_out_lumen(sample: Dict[str, Any]) -> bool:
        """Out-of-lumen rows carry no `tangents` column (no direction)."""

        return "tangents" not in sample

    @staticmethod
    def _num_directions(sample: Dict[str, Any]) -> int:
        """Number of tangent directions recorded for an on-centerline sample."""

        tangents = ast.literal_eval(sample["tangents"])
        if not tangents:
            return 0
        # Stored either as [[x,y,z], ...] or as a flat [x,y,z, ...].
        if isinstance(tangents[0], (list, tuple)):
            return len(tangents)
        return len(tangents) // 3

    @classmethod
    def _keep_bidirectional(cls, sample: Dict[str, Any]) -> bool:
        """SIRE is a bi-directional tracker, so the per-node `directions` field
        must be uniformly shape [2, 3] across the batch (a non-uniform field
        breaks DataLoader collation).

        - Out-of-lumen / augmentation rows: always kept. The sampler ignores
          their (absent) tangents and emits a 2-direction zero placeholder.
        - Direction rows (radius > 0): keep only those with exactly 2 tangents,
          discarding 1-direction endpoints and >= 3-direction bifurcations.
        """

        if cls._is_out_lumen(sample) or float(sample["radius"]) <= 0:
            return True
        return cls._num_directions(sample) == 2

    def _filter_bidirectional(self, rows: List[Dict], src: str) -> List[Dict]:

        kept = [s for s in rows if self._keep_bidirectional(s)]
        print(f"SIRECoWData[{src}]: kept {len(kept)}/{len(rows)} samples "
              f"(discarded != 2-direction direction rows).")
        return kept

    def load_samples(self, **kwds):
        """
        Load training samples following the original SIRE recipe: on-centerline
        samples (0.9) mixed with out-of-lumen samples (0.1, zeroed response).
        No off-centerline samples.
        """

        LOAD_LEN = self.trn_len if kwds["stage"] == "train" else self.val_len

        use_aug = kwds.get("augmentation", True)  # Use augmentation or not
        no_aug_df = pd.read_csv(self.sample_files["on_centerline"], sep=",")
        no_aug: List[Dict] = no_aug_df.to_dict(orient="records")
        no_aug = self._filter_bidirectional(no_aug, "on_centerline")

        if not use_aug or self.stage in ["val", "test"]:
            random.shuffle(no_aug)
            self.samples = no_aug[:LOAD_LEN]
            return

        out_lumen_df = pd.read_csv(self.sample_files["out_lumen"], sep=",")
        out_lumen: List[Dict] = out_lumen_df.to_dict(orient="records")
        random.shuffle(out_lumen)
        self.samples = no_aug[:int(0.9 * LOAD_LEN)] + \
            out_lumen[:int(0.1 * LOAD_LEN)]
        random.shuffle(self.samples)

    def prepare_data(self, sample: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Transform sample record to tensor for training.
        """

        case_name: str = sample["case"]
        # Out-of-lumen rows have no tangents and are always treated as
        # augmentation (zeroed sphere response), regardless of their radius.
        aug: bool = self._is_out_lumen(sample) or sample["radius"] <= 0

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
            "augmentation": aug,
            "tangents": ast.literal_eval(sample["tangents"]) if not aug else None,
        }
        data = self.load_transform(data)
        data = self.pre_transform(data)

        return data

    def _set_transform(self):

        # Keep LoadImageMask as a separate attribute (not inside the Compose):
        # SharedMemoryCleanUpCallback inspects `dataset.load_transform`, matching
        # the COACTCoWData / AdaSIRECoWData convention.
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
            SampleSIREImageOnly(
                npoints=self.config.npoints,
                subdivisions=self.config.subdivisions,
                alpha=self.config.alpha,
                r=self.config.beta,
            ),
        ])
