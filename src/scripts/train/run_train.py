"""Single-run training launcher driven by a CSV configuration row.

Usage:
    python src/scripts/run_train.py \
        --config-csv src/scripts/configs/train_configs.csv \
        --run-id asoca_baseline \
        --device cuda:0

Each row of the CSV produces an independent training run identified by
``run_id``. The run's log directory and the resolved CSV row are written to
``$DATA_ROOT/index/train_index.json`` so that test runs can locate the
checkpoint later via the same ID.
"""
import argparse
import glob
import json
import os
import re
from datetime import datetime

import torch

from src.scripts.dataset_configs.adasire_tracker_aorta24_train import AdaSIRETrackerAorta24TrainConfig
from src.scripts.dataset_configs.adasire_tracker_asoca_train import AdaSIRETrackerASOCATrainConfig
from src.scripts.dataset_configs.adasire_tracker_cow_train import AdaSIRETrackerCoWTrainConfig
from src.scripts.dataset_configs.adasire_tracker_parse_train import AdaSIRETrackerParseTrainConfig
from src.model.networks.gem_gcn import (
    GEMGCNAdaSIREMultiTaskScaleAverage,
    GEMGCNAdaSIREMultiTaskScaleFusion,
)
from src.model.losses.ce_loss import BCEFixedWeightedLoss, BCEGeoWeightedLoss
from src.scripts.common.run_utils import (
    LOGS_ROOT,
    TRAIN_INDEX_PATH,
    device_index,
    load_index,
    load_row,
    to_bool,
    to_float,
    to_int,
    to_list,
    update_index,
)
from src.trainer.pl_trainer_wrappers import PytorchLightningTrainWrapper


def _resolve_ckpt(spec: str, run_id: str):
    """Translate a ``ckpt`` CSV cell into a checkpoint path or ``None``.

    Accepted values:
      - empty / whitespace      -> ``None`` (fresh run; current default).
      - ``"latest"``            -> highest-step ``*.ckpt`` under the run's
                                  ``save_dir`` recorded in
                                  ``train_index.json``.
      - any other non-empty str -> treated as an explicit filesystem path.

    When non-None, ``PytorchLightningTrainWrapper`` (a) reuses the original
    versioned log directory (so events/checkpoints/metadata accumulate in
    place) and (b) calls ``trainer.fit(ckpt_path=...)`` to restore weights +
    optimizer/LR scheduler state + global step + epoch counters. Make sure
    the CSV row's ``max_epochs`` is larger than what the checkpoint already
    reached; otherwise Lightning will exit immediately.
    """

    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.lower() == "latest":
        idx = load_index(TRAIN_INDEX_PATH)
        entry = idx.get(run_id)
        if not entry:
            raise KeyError(
                f"Cannot resume run_id={run_id!r} with 'latest': no entry in "
                f"{TRAIN_INDEX_PATH}. Run a fresh training first.")
        ckpt_dir = os.path.join(entry["save_dir"], "checkpoints")
        ckpts = glob.glob(os.path.join(ckpt_dir, "*step=*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        return max(ckpts,
                   key=lambda p: int(re.search(r"step=(\d+)", p).group(1)))
    if not os.path.exists(spec):
        raise FileNotFoundError(f"ckpt path does not exist: {spec}")
    return spec


_TRAIN_CONFIGS = {
    "asoca": AdaSIRETrackerASOCATrainConfig,
    "aorta24": AdaSIRETrackerAorta24TrainConfig,
    "cow": AdaSIRETrackerCoWTrainConfig,
    "parse": AdaSIRETrackerParseTrainConfig,
}

# Ablation selectors (CSV columns ``scale_fusion`` / ``dir_loss``).
# Defaults below match the hard-coded baseline so CSVs without these columns
# behave exactly as before.
_SCALE_FUSIONS = {
    "gating": GEMGCNAdaSIREMultiTaskScaleFusion,  # ScaleMLPFusionLayer (gated)
    "mean": GEMGCNAdaSIREMultiTaskScaleAverage,   # ScaleSimpleMeanLayer (no gate)
}
_DIR_LOSSES = {
    "geo": BCEGeoWeightedLoss,      # target*pos_w + haversine dists
    "fixed": BCEFixedWeightedLoss,  # target*pos_w + (1-target)*neg_w (no geo)
}


def _build_config(row: dict, device: str, run_id: str):

    dataset = row["dataset"].strip().lower()
    if dataset not in _TRAIN_CONFIGS:
        raise ValueError(
            f"Unknown dataset {dataset!r} for run {run_id!r}. "
            f"Choices: {sorted(_TRAIN_CONFIGS)}")

    config = _TRAIN_CONFIGS[dataset]()
    config.run_id = run_id
    config.device = device
    config.trainer["devices"] = [device_index(device)]

    config.base_radius = to_float(row["base_radius"])
    config.radius_param = (row.get("radius_param", "") or "logit").strip().lower()
    r_ref_cell = row.get("r_ref", "").strip()
    config.r_ref = to_float(r_ref_cell) if r_ref_cell else None
    if config.radius_param == "log" and config.r_ref is None:
        raise ValueError(
            f"run_id={run_id!r}: radius_param='log' requires r_ref to be set "
            f"in the CSV row.")
    config.pl_config["radius_param"] = config.radius_param
    config.pl_config["r_ref"] = config.r_ref
    config.min_scale = to_float(row["min_scale"])
    config.max_scale = to_float(row["max_scale"])
    config.max_num_scales = to_int(row["max_num_scales"])
    config.rand_scales = to_bool(row["rand_scales"])
    config.fix_scales = to_list(row["fix_scales"])

    config.pl_config["loss_params"]["pos_weight"] = to_float(row["pos_weight"])
    config.pl_config["direction_loss_alpha"] = to_float(
        row["direction_loss_alpha"])

    # Ablation axes (default to the baseline: gated fusion + geo-weighted loss).
    sf = (row.get("scale_fusion", "") or "gating").strip().lower()
    dl = (row.get("dir_loss", "") or "geo").strip().lower()
    if sf not in _SCALE_FUSIONS:
        raise ValueError(
            f"run_id={run_id!r}: unknown scale_fusion {sf!r}; "
            f"choices: {sorted(_SCALE_FUSIONS)}")
    if dl not in _DIR_LOSSES:
        raise ValueError(
            f"run_id={run_id!r}: unknown dir_loss {dl!r}; "
            f"choices: {sorted(_DIR_LOSSES)}")
    config.pl_config["model"] = _SCALE_FUSIONS[sf]
    config.pl_config["loss"] = _DIR_LOSSES[dl]
    config.pl_config["loss_params"]["neg_weight"] = to_float(
        row.get("neg_weight", "") or "1")

    if row.get("seed", "").strip():
        config.pl_config["seed"] = to_int(row["seed"])

    cap_cell = row.get("max_train_samples", "").strip()
    if cap_cell:
        config.max_train_samples = to_int(cap_cell)

    config.trainer["max_epochs"] = to_int(row["max_epochs"])
    config.trainer["accumulate_grad_batches"] = to_int(
        row["accumulate_grad_batches"])
    config.batch_size = to_int(row["batch_size"])
    config.effective_bs = (
        config.batch_size * config.trainer["accumulate_grad_batches"])
    config.pl_config["batch_size"] = config.batch_size

    config.log_dir = LOGS_ROOT
    config.trainer["default_root_dir"] = LOGS_ROOT
    config.ckpt = _resolve_ckpt(row.get("ckpt", ""), run_id)

    mix_cells = [row.get(c, "").strip() for c in ("mix_on", "mix_off", "mix_out")]
    provided = [c for c in mix_cells if c]
    if provided:
        if len(provided) != 3:
            raise ValueError("mix_on/mix_off/mix_out must be set together")
        mix = {
            "on_centerline": float(mix_cells[0]),
            "off_centerline": float(mix_cells[1]),
            "out_lumen": float(mix_cells[2]),
        }
        assert abs(sum(mix.values()) - 1.0) < 1e-6, \
            f"mix_on/off/out must sum to 1.0 (got {mix})"
        config.sample_mix = mix

    aim = row.get("aim", "").strip()
    if aim:
        config.aim = aim
    config.aim = f"[run_id={run_id}] {getattr(config, 'aim', '')}".strip()

    return config


def main():
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-csv", required=True,
        help="Path to a CSV file containing training run configurations.")
    parser.add_argument(
        "--run-id", required=True,
        help="Value of the ``run_id`` column to execute from the CSV.")
    parser.add_argument(
        "--device", default="cuda:0",
        help="Training device, e.g. ``cuda:0``.")
    args = parser.parse_args()

    row = load_row(args.config_csv, args.run_id)
    config = _build_config(row, args.device, args.run_id)

    print(f"[run_train] starting run_id={args.run_id} dataset={row['dataset']} "
          f"device={args.device}")
    print(f"[run_train] config row: {row}")

    wrapper = PytorchLightningTrainWrapper(config=config)
    log_dir = wrapper.pl_trainer.logger.log_dir
    os.makedirs(log_dir, exist_ok=True)

    metadata = {
        "run_id": args.run_id,
        "csv_row": row,
        "csv_path": os.path.abspath(args.config_csv),
        "device": args.device,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "log_dir": log_dir,
        "resumed_from": config.ckpt,
    }
    with open(os.path.join(log_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    update_index(TRAIN_INDEX_PATH, args.run_id, log_dir, row,
                 extra={"dataset": row["dataset"]})

    wrapper.train()

    finished = {"finished_at": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(log_dir, "metadata.json")) as f:
        metadata = json.load(f)
    metadata.update(finished)
    with open(os.path.join(log_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[run_train] finished run_id={args.run_id} -> {log_dir}")


if __name__ == "__main__":
    main()
