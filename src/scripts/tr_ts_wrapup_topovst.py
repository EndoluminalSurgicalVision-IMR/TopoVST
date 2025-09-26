import os
import ast
import json
import glob
from argparse import ArgumentParser

import torch
import numpy as np

from src.scripts.adasire_tracker_aorta24_train import AdaSIRETrackerAorta24TrainConfig
from src.scripts.adasire_tracker_asoca_train import AdaSIRETrackerASOCATrainConfig
from src.scripts.adasire_tracker_aorta24_test import AdaSIRETrackerAorta24TestConfig
from src.scripts.adasire_tracker_asoca_test import AdaSIRETrackerASOCATestConfig
from src.trainer.pl_trainer_wrappers import PytorchLightningTrainWrapper
from src.tester.tracking_test_wrappers import SeedBasedTrackingWrapper
from src.tester.wavefront_tracking_pipelines import AdaSireWaveFrontFixToSkeletonSeeds


if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    parser = ArgumentParser()
    parser.add_argument("--device", help="train/testing devices.", type=str,
                        default="cuda:0")
    parser.add_argument("--use_steps", help="A list of training steps to use.",
                        type=int, nargs='+')

    args = parser.parse_args()
    print("Use device: ", args.device)
    print("Use model steps: ", args.use_steps)

    # NOTE: To perform only testing part, just comment the following code
    tr_config = AdaSIRETrackerASOCATrainConfig()
    # NOTE: You can adjust here the training settings
    tr_config.device = args.device
    tr_config.trainer["devices"] = [ast.literal_eval(args.device[-1])]
    tr_config.trainer["max_epochs"] = 2
    tr_config.rand_scales = True
    tr_config.fix_scales = []
    tr_config.pl_config["loss_params"]["pos_weight"] = 10.0  # Ablation term
    tr_config.pl_config["direction_loss_alpha"] = 5.0  # d loss balance term
    tr_config.base_radius = 10.0
    tr_config.max_num_scales = 15
    tr_config.ckpt = None

    trainer = PytorchLightningTrainWrapper(config=tr_config)
    trainer.train()

    # NOTE: To perform only training part, just comment the following code
    os.system("clear")
    # Test
    save_dir = trainer.pl_trainer.logger.log_dir
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    ts_config = AdaSIRETrackerASOCATestConfig()
    ts_config.device = args.device
    # NOTE: You can adjust here the testing settings
    ts_config.scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
