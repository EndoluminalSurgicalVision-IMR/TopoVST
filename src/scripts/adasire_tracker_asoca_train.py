from ast import literal_eval

import torch

from src.utils.config import BaseConfig
from src.dataset.adasire_datasets import AdaSIREASOCAData
from src.model.adasire.ada_sire_trackers import AdaSIREMultiTaskScaleFusionTracker
from src.model.networks.gem_gcn import GEMGCNAdaSIREMultiTaskScaleFusion
from src.model.losses.ce_loss import BCEGeoWeightedLoss
from src.trainer.pl_trainer_wrappers import PytorchLightningTrainWrapper


class AdaSIRETrackerASOCATrainConfig(BaseConfig):

    # Specify your training purpose
    aim = ("Baseline model.")

    device = "cuda:0"

    rand_scales = True  # Use randomly generated sampling scales or not
    fix_scales = []
    min_num_scales = 3
    max_num_scales = 15
    min_scale = 0.2
    max_scale = 15.0
    npoints = 64  # Feature length per node
    subdivisions = 3  # 642 points
    augmentation = True  # Enable off-centerline samples in training
    base_radius = 5.0  # Unit: mm

    dataset = AdaSIREASOCAData
    dataset_name = "ASOCA"
    src_dir = ""  # User-defined
    # NOTE: sample CSV files should be generated in advance!
    samples = {
        "train": {
            "on_centerline": "tmp_data/samples/ASOCA_train_on_centerline_raware_multidir.csv",
            "off_centerline": "tmp_data/samples/ASOCA_train_off_centerline_raware_multidir.csv",
            "out_lumen": "tmp_data/samples/ASOCA_train_out_lumen.csv",
        },
        "val": {
            "on_centerline": "tmp_data/samples/ASOCA_val_on_centerline_raware_multidir.csv",
        }
    }
    window_level = 200
    window_width = 600
    train_samples = 20000
    val_samples = 500

    task_name = "AdaSIRETracker"
    log_dir = "tmp_data/logs"
    trainer = {
        "accelerator": "gpu",
        "devices": [literal_eval(device[-1])],
        "benchmark": False,  # cuDNN deterministic algorithm for reproducibility
        "check_val_every_n_epoch": 1,
        "num_sanity_val_steps": 0,
        "default_root_dir": log_dir,
        "enable_checkpointing": True,
        "log_every_n_steps": 1,
        "max_epochs": 10,
        "accumulate_grad_batches": 2,
    }

    batch_size = 4
    num_workers = 8
    effective_bs = batch_size * trainer["accumulate_grad_batches"]

    # Below are settings for pytorch_lightning model
    pl_module = AdaSIREMultiTaskScaleFusionTracker
    pl_config = {
        "subdivisions": subdivisions,
        "model": GEMGCNAdaSIREMultiTaskScaleFusion,
        "model_params": {
            "convs": 3,
            "in_features": 64,
            "embed_dim": 128,
            "out_channels": 1,
        },
        "seed": 4294967295,  # Default value used by monai
        "optimizer": torch.optim.Adam,
        "optimizer_params": {
            "lr": 0.0005,
        },
        "lr_scheduler": torch.optim.lr_scheduler.StepLR,
        "lr_scheduler_params": {
            'step_size': 1,
            'gamma': 0.99
        },
        "loss": BCEGeoWeightedLoss,
        "loss_params": {
            "subdivisions": 3,
            "pos_weight": 10.0,
        },
        "direction_loss_alpha": 5.0,
        "batch_size": batch_size,
    }
    ckpt = None  # Set to None to disable checkpoint loading.


if __name__ == "__main__":

    config = AdaSIRETrackerASOCATrainConfig()
    trainer = PytorchLightningTrainWrapper(config=config)
    trainer.train()
