"""Single-run tracking-test launcher driven by a CSV configuration row.

Usage:
    python src/scripts/run_test.py \
        --config-csv src/scripts/configs/test_configs.csv \
        --run-id cow_baseline_s200_cyc30 \
        --device cuda:0

The CSV row references a previously executed training run via
``train_run_id``; its log directory and dataset are read from
``$DATA_ROOT/index/train_index.json``. Test results are written under
``$DATA_ROOT/tests/<train_run_id>/<test_run_id>/<timestamp>/`` and the run
is registered in ``$DATA_ROOT/index/test_index.json`` for analysis.
"""
import argparse
import glob
import json
import os
from datetime import datetime

import torch

from src.scripts.dataset_configs.adasire_tracker_aorta24_test import AdaSIRETrackerAorta24TestConfig
from src.scripts.dataset_configs.adasire_tracker_asoca_test import AdaSIRETrackerASOCATestConfig
from src.scripts.dataset_configs.adasire_tracker_cow_test import AdaSIRETrackerCoWTestConfig
from src.scripts.dataset_configs.adasire_tracker_parse_test import AdaSIRETrackerParseTestConfig
from src.eval import evaluate_dir, evaluate_dir_from_mask
from src.postprocess import METHODS as POSTPROCESS_METHODS
from src.postprocess.runner import run_post_process
from src.postprocess.to_mask import rasterize_dir
from src.scripts.common.run_utils import (
    TESTS_ROOT,
    TEST_INDEX_PATH,
    TRAIN_INDEX_PATH,
    load_index,
    load_row,
    to_bool,
    to_float,
    to_int,
    to_list,
    update_index,
)
from src.tester.tracking_test_wrappers import SeedBasedTrackingWrapper
from src.tester.wavefront_tracking_pipelines import (
    AdaSireWaveFrontFixSeeds,
    AdaSireWaveFrontFixToSkeletonSeeds,
    AdaSireWaveFrontSkeletonSeeds,
)


_TEST_CONFIGS = {
    "asoca": AdaSIRETrackerASOCATestConfig,
    "aorta24": AdaSIRETrackerAorta24TestConfig,
    "cow": AdaSIRETrackerCoWTestConfig,
    "parse": AdaSIRETrackerParseTestConfig,
}

_SEED_PIPELINES = {
    "fix": AdaSireWaveFrontFixSeeds,
    "skeleton": AdaSireWaveFrontSkeletonSeeds,
    "fix_to_skeleton": AdaSireWaveFrontFixToSkeletonSeeds,
}


def _resolve_train_run(train_run_id: str):

    train_idx = load_index(TRAIN_INDEX_PATH)
    if train_run_id not in train_idx:
        raise KeyError(
            f"train_run_id {train_run_id!r} not found in {TRAIN_INDEX_PATH}. "
            f"Run training for it first via run_train.py.")
    entry = train_idx[train_run_id]
    log_dir = entry["save_dir"]
    dataset = (entry.get("dataset")
               or entry.get("config_row", {}).get("dataset"))
    if not dataset:
        raise ValueError(
            f"Train index entry for {train_run_id!r} is missing a dataset.")
    train_row = entry.get("config_row", {}) or {}
    return log_dir, dataset.strip().lower(), train_row


def _find_checkpoint(log_dir: str, step: int) -> str:

    ckpt_dir = os.path.join(log_dir, "checkpoints")
    matches = sorted(glob.glob(os.path.join(ckpt_dir, f"*step={step}.ckpt")))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching step={step} in {ckpt_dir}. "
            f"Available: {sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else 'missing dir'}")
    return matches[0]


def _build_config(row: dict, dataset: str, mdl_ckpt: str, device: str,
                  run_id: str):

    if dataset not in _TEST_CONFIGS:
        raise ValueError(
            f"Unknown dataset {dataset!r} for run {run_id!r}. "
            f"Choices: {sorted(_TEST_CONFIGS)}")

    config = _TEST_CONFIGS[dataset]()
    config.run_id = run_id
    config.device = device

    config.scales = to_list(row["test_scales"])
    config.infer_config["max_seeds"] = to_int(row["max_seeds"])
    config.infer_config["max_front_len"] = to_int(row["max_front_len"])
    config.infer_config["base_radius"] = to_float(row["base_radius"])
    radius_param = (row.get("radius_param", "") or "logit").strip().lower()
    r_ref_cell = row.get("r_ref", "").strip()
    config.infer_config["radius_param"] = radius_param
    config.infer_config["r_ref"] = (
        to_float(r_ref_cell) if r_ref_cell else None)
    if radius_param == "log" and config.infer_config["r_ref"] is None:
        raise ValueError(
            f"run_id={run_id!r}: radius_param='log' requires r_ref to be set "
            f"in the test CSV row.")
    config.min_cycle_length_mm = to_float(row["min_cycle_length_mm"])

    config.mdl_ckpt = mdl_ckpt
    config.ckpt_time = mdl_ckpt.split("/")[-3]
    config.ckpt_name = mdl_ckpt.split("/")[-1]

    return config


def _pick_pipeline(row: dict):

    name = (row.get("seed_strategy") or "fix_to_skeleton").strip().lower()
    if name not in _SEED_PIPELINES:
        raise ValueError(
            f"Unknown seed_strategy {name!r}. "
            f"Choices: {sorted(_SEED_PIPELINES)}")
    return _SEED_PIPELINES[name]


def main():
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-csv", required=True,
        help="Path to a CSV file containing testing run configurations.")
    parser.add_argument(
        "--run-id", required=True,
        help="Value of the ``run_id`` column to execute from the CSV.")
    parser.add_argument(
        "--device", default="cuda:0",
        help="Inference device, e.g. ``cuda:0``.")
    args = parser.parse_args()

    row = load_row(args.config_csv, args.run_id)
    train_run_id = row["train_run_id"].strip()
    train_log_dir, dataset, train_row = _resolve_train_run(train_run_id)
    mdl_ckpt = _find_checkpoint(train_log_dir, to_int(row["step"]))

    test_rparam = (row.get("radius_param", "") or "logit").strip().lower()
    train_rparam = (train_row.get("radius_param", "") or "logit").strip().lower()
    if test_rparam != train_rparam:
        raise ValueError(
            f"radius_param mismatch: train run {train_run_id!r} was trained "
            f"with {train_rparam!r}, but test row {args.run_id!r} requests "
            f"{test_rparam!r}. The radius head decoding would be wrong.")
    if test_rparam == "log":
        test_rref = (row.get("r_ref", "") or "").strip()
        train_rref = (train_row.get("r_ref", "") or "").strip()
        if test_rref != train_rref:
            raise ValueError(
                f"r_ref mismatch under radius_param='log': train={train_rref!r} "
                f"vs test={test_rref!r}. Predicted radii would be off by a "
                f"constant scale factor.")

    config = _build_config(row, dataset, mdl_ckpt, args.device, args.run_id)

    pp_methods = [m.strip().lower()
                  for m in (row.get("post_process", "") or "").split(",")
                  if m.strip() and m.strip().lower() != "none"]
    for m in pp_methods:
        base = m.split(":", 1)[0]  # allow "name:value" param overrides
        if base not in POSTPROCESS_METHODS:
            raise ValueError(
                f"Unknown post_process method {base!r}; "
                f"choices: {sorted(POSTPROCESS_METHODS)}")
    do_eval = to_bool(row.get("evaluate", "true") or "true")
    finalize = (row.get("finalize", "") or "").strip().lower()
    if finalize and finalize not in {"to_mask"}:
        raise ValueError(
            f"Unknown finalize value {finalize!r}; choices: ['', 'to_mask']")

    save_dir = os.path.join(
        TESTS_ROOT, train_run_id, args.run_id,
        datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)

    print(f"[run_test] starting run_id={args.run_id} dataset={dataset} "
          f"device={args.device}")
    print(f"[run_test] checkpoint: {mdl_ckpt}")
    print(f"[run_test] save_dir:   {save_dir}")
    print(f"[run_test] config row: {row}")

    with open(os.path.join(save_dir, "configs.json"), "w") as f:
        json.dump(config.print_self(), f, indent=4, default=str)
    with open(os.path.join(save_dir, "readme.txt"), "w") as f:
        f.write(config.get_readme())
    metadata = {
        "run_id": args.run_id,
        "train_run_id": train_run_id,
        "dataset": dataset,
        "csv_row": row,
        "csv_path": os.path.abspath(args.config_csv),
        "device": args.device,
        "ckpt": mdl_ckpt,
        "train_log_dir": train_log_dir,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "save_dir": save_dir,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    update_index(TEST_INDEX_PATH, args.run_id, save_dir, row,
                 extra={
                     "train_run_id": train_run_id,
                     "dataset": dataset,
                     "post_process_methods": pp_methods,
                     "evaluate": do_eval,
                     "finalize": finalize,
                 })

    pipeline_cls = _pick_pipeline(row)
    pipeline = pipeline_cls(config, save_dir)
    tester = SeedBasedTrackingWrapper(config, pipeline, save_dir)

    parallel = to_int(row.get("parallel_processes") or "1")
    if parallel > 1:
        tester.test_parallel_wrapper(processes=parallel)
    else:
        tester.test()

    case_to_seg_path = {case_id: pred for _, case_id, pred in tester.cases}
    pp_dirs = []
    for method_name in pp_methods:
        pp_dir = os.path.join(save_dir, f"pp_{method_name}")
        print(f"[run_test] post_process: {method_name} -> {pp_dir}")
        run_post_process(method_name, raw_dir=save_dir, out_dir=pp_dir,
                         case_to_seg_path=case_to_seg_path)
        pp_dirs.append(pp_dir)

    eval_csvs = []
    if do_eval:
        for target_dir in [save_dir] + pp_dirs:
            eval_out = os.path.join(target_dir, "eval")
            print(f"[run_test] evaluate: {target_dir} -> {eval_out}")
            csv_path = evaluate_dir(
                pred_dir=target_dir, dataset=dataset, out_dir=eval_out)
            eval_csvs.append(csv_path)

    mask_dirs = []
    eval_mask_csvs = []
    if finalize == "to_mask":
        for target_dir in [save_dir] + pp_dirs:
            mask_dir = os.path.join(target_dir, "mask")
            print(f"[run_test] finalize=to_mask: {target_dir} -> {mask_dir}")
            rasterize_dir(in_dir=target_dir, out_dir=mask_dir,
                          case_to_seg_path=case_to_seg_path)
            mask_dirs.append(mask_dir)
            if do_eval:
                eval_mask_out = os.path.join(target_dir, "eval_mask")
                print(f"[run_test] evaluate_mask: {mask_dir} -> {eval_mask_out}")
                csv_path = evaluate_dir_from_mask(
                    mask_dir=mask_dir, dataset=dataset, out_dir=eval_mask_out)
                eval_mask_csvs.append(csv_path)

    with open(os.path.join(save_dir, "metadata.json")) as f:
        metadata = json.load(f)
    metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
    metadata["post_process_methods"] = pp_methods
    metadata["post_process_dirs"] = pp_dirs
    metadata["evaluate"] = do_eval
    metadata["eval_csvs"] = eval_csvs
    metadata["finalize"] = finalize
    metadata["mask_dirs"] = mask_dirs
    metadata["eval_mask_csvs"] = eval_mask_csvs
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[run_test] finished run_id={args.run_id} -> {save_dir}")


if __name__ == "__main__":
    main()
