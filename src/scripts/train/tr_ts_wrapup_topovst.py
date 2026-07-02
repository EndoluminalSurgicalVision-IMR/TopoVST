import os
import ast
import json
import glob
from argparse import ArgumentParser

import torch
import numpy as np

from src.scripts.dataset_configs.adasire_tracker_aorta24_train import AdaSIRETrackerAorta24TrainConfig
from src.scripts.dataset_configs.adasire_tracker_asoca_train import AdaSIRETrackerASOCATrainConfig
from src.scripts.dataset_configs.adasire_tracker_cow_train import AdaSIRETrackerCoWTrainConfig
from src.scripts.dataset_configs.adasire_tracker_aorta24_test import AdaSIRETrackerAorta24TestConfig
from src.scripts.dataset_configs.adasire_tracker_asoca_test import AdaSIRETrackerASOCATestConfig
from src.scripts.dataset_configs.adasire_tracker_cow_test import AdaSIRETrackerCoWTestConfig
from src.trainer.pl_trainer_wrappers import PytorchLightningTrainWrapper
from src.tester.tracking_test_wrappers import SeedBasedTrackingWrapper
from src.tester.wavefront_tracking_pipelines import AdaSireWaveFrontFixToSkeletonSeeds


_TRAIN_CONFIGS = {
    "asoca": AdaSIRETrackerASOCATrainConfig,
    "aorta24": AdaSIRETrackerAorta24TrainConfig,
    "cow": AdaSIRETrackerCoWTrainConfig,
}

_TEST_CONFIGS = {
    "asoca": AdaSIRETrackerASOCATestConfig,
    "aorta24": AdaSIRETrackerAorta24TestConfig,
    "cow": AdaSIRETrackerCoWTestConfig,
}

# Per-dataset run-time overrides. Each entry tunes the wrapup defaults to the
# physical vessel scale of the dataset (training scale range, test scale grid,
# and base radius used for both training graph construction and inference).
# CoW values follow the measured mis_radius distribution (p50≈1.05 mm,
# p99≈2.17 mm), so the test grid is denser at sub-mm radii than ASOCA's.
_DATASET_OVERRIDES = {
    "asoca": {
        "base_radius": 10.0,
        "min_scale": 0.2,
        "max_scale": 15.0,
        "max_num_scales": 15,
        "test_scales": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    },
    "aorta24": {
        "base_radius": 30.0,
        "min_scale": 1.0,
        "max_scale": 30.0,
        "max_num_scales": 15,
        "test_scales": [2, 4, 6, 10, 15, 20, 25, 30, 40, 50],
    },
    "cow": {
        "base_radius": 5.0,
        "min_scale": 0.1,
        "max_scale": 6.0,
        "max_num_scales": 15,
        "test_scales": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    },
}


if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    parser = ArgumentParser()
    parser.add_argument("--device", help="train/testing devices.", type=str,
                        default="cuda:0")
    parser.add_argument("--use_steps", help="A list of training steps to use.",
                        type=int, nargs='+')
    parser.add_argument("--dataset", help="Dataset to use.", type=str,
                        default="asoca",
                        choices=list(_TRAIN_CONFIGS.keys()))

    args = parser.parse_args()
    print("Use device: ", args.device)
    print("Use dataset: ", args.dataset)
    print("Use model steps: ", args.use_steps)

    # NOTE: To perform only testing part, just comment the following code
    tr_config = _TRAIN_CONFIGS[args.dataset]()
    overrides = _DATASET_OVERRIDES[args.dataset]
    # NOTE: You can adjust here the training settings
    tr_config.device = args.device
    tr_config.trainer["devices"] = [ast.literal_eval(args.device[-1])]
    tr_config.trainer["max_epochs"] = 2
    tr_config.rand_scales = True
    tr_config.fix_scales = []
    tr_config.pl_config["loss_params"]["pos_weight"] = 10.0  # Ablation term
    tr_config.pl_config["direction_loss_alpha"] = 5.0  # d loss balance term
    tr_config.base_radius = overrides["base_radius"]
    tr_config.min_scale = overrides["min_scale"]
    tr_config.max_scale = overrides["max_scale"]
    tr_config.max_num_scales = overrides["max_num_scales"]
    tr_config.ckpt = None

    trainer = PytorchLightningTrainWrapper(config=tr_config)
    trainer.train()

    # NOTE: To perform only training part, just comment the following code
    os.system("clear")
    # Test
    save_dir = trainer.pl_trainer.logger.log_dir
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    ts_config = _TEST_CONFIGS[args.dataset]()
    ts_config.device = args.device
    # NOTE: You can adjust here the testing settings
    ts_config.scales = overrides["test_scales"]
    ts_config.infer_config["max_seeds"] = 1000  # Use only one seed
    ts_config.infer_config["max_front_len"] = 20  # Use a large max front len
    ts_config.infer_config["base_radius"] = tr_config.base_radius

    mdl_ckpts = []
    for step in args.use_steps:
        paths = glob.glob(f"*step={step}.ckpt", root_dir=ckpt_dir)
        if not paths:
            continue
        mdl_ckpts.append(paths[0])
    mdl_ckpts = [os.path.join(ckpt_dir, ckpt_name) for ckpt_name in mdl_ckpts]

    for mdl_ckpt in mdl_ckpts:
        # Specify the model checkpoint to use
        ts_config.mdl_ckpt = mdl_ckpt
        ts_config.ckpt_time = mdl_ckpt.split("/")[-3]
        ts_config.ckpt_name = mdl_ckpt.split("/")[-1]
        # Create test result folders based on model checkpoints
        save_location = ts_config.get_save_location()
        if not os.path.exists(save_location):
            os.makedirs(save_location)
        with open(os.path.join(save_location, "configs.json"), "w") as f:
            json.dump(ts_config.print_self(), f, indent=4)
        with open(os.path.join(save_location, "readme.txt"), "w") as f:
            f.write(ts_config.get_readme())
        pipeline = AdaSireWaveFrontFixToSkeletonSeeds(ts_config, save_location)
        tester = SeedBasedTrackingWrapper(ts_config, pipeline, save_location)
        tester.test_parallel_wrapper(processes=2)

        del pipeline
        del tester
        torch.cuda.empty_cache()
