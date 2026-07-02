from ast import literal_eval

import torch

from src.utils.config import BaseConfig
from src.dataset.adasire_datasets import AdaSIRECoWData
from src.model.adasire.ada_sire_trackers import AdaSIREMultiTaskScaleFusionTracker
from src.model.networks.gem_gcn import GEMGCNAdaSIREMultiTaskScaleFusion
from src.model.losses.ce_loss import BCEGeoWeightedLoss
from src.trainer.pl_trainer_wrappers import PytorchLightningTrainWrapper


class AdaSIRETrackerCoWTrainConfig(BaseConfig):

    aim = ("Baseline model for CoW (Circle of Willis).")

    device = "cuda:0"

    rand_scales = True
    fix_scales = []
    min_num_scales = 3
    max_num_scales = 15
    # CoW vessel radii are small (mostly < 2 mm); keep a tight scale range.
    min_scale = 0.2
    max_scale = 15.0
    npoints = 64
    subdivisions = 3  # 642 points
    augmentation = True
    base_radius = 5.0  # Unit: mm

    dataset = AdaSIRECoWData
    dataset_name = "CoW"
    # Dataset033_BinTopCoW24_CT root (contains imagesTr/Ts, labelsTr/Ts).
    src_dir = ""
    # NOTE: sample CSV files should be generated in advance!
    # NOTE: The "val" CSV needs to be produced by the user from the training
    # cases (e.g., reserving ~10% of cases). Pointing at a non-existent file
    # mirrors the ASOCA/Aorta24 convention and forces an explicit split.
    samples = {
        "train": {
            "on_centerline": "",
            "off_centerline": "",
            "out_lumen": "",
        },
        # Val loop is disabled via trainer["limit_val_batches"]=0, but the
        # PL wrapper still instantiates a val dataset. Point at the train CSV
        # so dataset construction succeeds; no batches will be iterated.
        "val": {
            "on_centerline": "",
        }
    }
    window_level = 200
    window_width = 600
    train_samples = 20000  # legacy; training now consumes all on-centerline rows
    val_samples = 500
    # Optional per-epoch cap on the number of training samples. 0 = uncapped
    # (consume all on-centerline rows, scaling off/out by sample_mix). When set
    # to a positive value via the CSV `max_train_samples` column, the per-epoch
    # total is capped at that value with the on/off/out mix preserved.
    max_train_samples = 0

    # Per-epoch composition of training samples. on/off/out must sum to 1.
    # Epoch size = len(on_centerline_csv) / sample_mix["on_centerline"].
    sample_mix = {"on_centerline": 0.6, "off_centerline": 0.3, "out_lumen": 0.1}

    task_name = "AdaSIRETracker"
    log_dir = ""
    trainer = {
        "accelerator": "gpu",
        "devices": [literal_eval(device[-1])],
        "benchmark": False,
        "check_val_every_n_epoch": 1,
        "num_sanity_val_steps": 0,
        "limit_val_batches": 0,
        "default_root_dir": log_dir,
        "enable_checkpointing": True,
        "log_every_n_steps": 1,
        "max_epochs": 10,
        "max_steps": 20000,
        "accumulate_grad_batches": 2,
    }

    batch_size = 4
    num_workers = 8
    effective_bs = batch_size * trainer["accumulate_grad_batches"]

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
        "seed": 4294967295,
        "optimizer": torch.optim.Adam,
        "optimizer_params": {
            "lr": 0.0005,
        },
        "lr_scheduler": torch.optim.lr_scheduler.StepLR,
        "lr_scheduler_params": {
            "step_size": 1,
            "gamma": 0.99,
        },
        "loss": BCEGeoWeightedLoss,
        "loss_params": {
            "subdivisions": 3,
            "pos_weight": 10.0,
        },
        "direction_loss_alpha": 5.0,
        "batch_size": batch_size,
    }
    ckpt = None


if __name__ == "__main__":

    config = AdaSIRETrackerCoWTrainConfig()
    trainer = PytorchLightningTrainWrapper(config=config)
    trainer.train()
