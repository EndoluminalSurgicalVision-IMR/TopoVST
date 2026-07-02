import os
import json
from datetime import datetime

from src.utils.config import BaseConfig
from src.model.adasire.ada_sire_trackers import AdaSIREMultiTaskScaleFusionTracker
from src.tester.wavefront_tracking_pipelines import AdaSireWaveFrontSkeletonSeeds
from src.tester.tracking_test_wrappers import SeedBasedTrackingWrapper


class AdaSIRETrackerCoWTestConfig(BaseConfig):

    device = ""  # Place-holder

    dataset_name = "CoW"
    # nnUNet-style splits json with "train"/"val"/"test" lists of case IDs.
    splits = ""  # User-defined
    src_dir = ""
    preds = ""  # Segmentation predictions directory
    phase = "test"

    seeds_folder = ""  # User-defined, should match the seeds generated for the test set by generate_seeds.py

    WINDOW_LEVEL = 200
    WINDOW_WIDTH = 600

    scales = []
    npoints = 64
    subdivisions = 3

    # Allow cycle closure for CoW (Circle of Willis loop perimeter ~50-100 mm).
    # Cycles whose geometric length is below this threshold are still rejected
    # to suppress spurious small loops from wavefront oscillations.
    min_cycle_length_mm = 30.0

    pl_module = AdaSIREMultiTaskScaleFusionTracker
    ckpt_time = ""
    ckpt_name = ""
    mdl_ckpt = ""

    infer_config = {
        "max_front_len": 20,
        "normalization": "sigmoid",
        "max_seeds": 20,
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

    config = AdaSIRETrackerCoWTestConfig()
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
