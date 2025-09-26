import os
import json
from datetime import datetime

import numpy as np

from src.utils.config import BaseConfig
from src.model.adasire.ada_sire_trackers import AdaSIREMultiTaskScaleFusionTracker
from src.tester.wavefront_tracking_pipelines import AdaSireWaveFrontFixSeeds, AdaSireWaveFrontSkeletonSeeds
from src.tester.tracking_test_wrappers import SeedBasedTrackingWrapper


class AdaSIRETrackerASOCATestConfig(BaseConfig):

    device = ""  # Place-holder

    dataset_name = "ASOCA"
    splits = ""  # Train-Val-Test split file in nnUNet style
    src_dir = ""  # User-defined
    preds = ""  # Segmentation predictions
    phase = "test"  # Which part of data you want to use

    # Seed configuration
    seeds_folder = ""  # or None, if not using pre-generated seeds

    # CT windowing configuration
    WINDOW_LEVEL = 200
    WINDOW_WIDTH = 600

    # Use testing settings
    scales = []
    npoints = 64
    subdivisions = 3  # 642 points

    pl_module = AdaSIREMultiTaskScaleFusionTracker
    ckpt_time = ""
    ckpt_name = ""
    mdl_ckpt = ""

    infer_config = {
        "max_front_len": 20,
        "normalization": "sigmoid",
        "max_seeds": 1000,
        "base_radius": 5.0,
        "stopping": {},
    }

    def get_save_location(self):

        base_dir = os.path.join(*self.mdl_ckpt.split("/")[:-2])
        header = os.path.join(base_dir, f'test_{self.ckpt_name.split(".")[0]}')
        tail = datetime.now().strftime("%Y%m%d_%H%M%S")

        return os.path.join(header, tail)

    def get_readme(self):

        readme = (
            f"Test on scales {self.scales} with "
            f"base radius {self.infer_config['base_radius']}."
        )

        return readme


if __name__ == "__main__":

    config = AdaSIRETrackerASOCATestConfig()
    save_location = config.get_save_location()
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    with open(os.path.join(save_location, "configs.json"), "w") as f:
        json.dump(config.print_self(), f, indent=4)
    with open(os.path.join(save_location, "readme.txt"), "w") as f:
        f.write(config.get_readme())

    pipeline = AdaSireWaveFrontSkeletonSeeds(config, save_location)
    tester = SeedBasedTrackingWrapper(config, pipeline, save_location)
    tester.test()
