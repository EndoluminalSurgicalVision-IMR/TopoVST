"""Measure the radius distribution of CoW centerline VTPs.

Reads the per-cell radius arrays (``mis_radius``, ``ce_radius``,
``voreen_radius``) from CoW VTP files without modifying them, and writes
summary statistics plus histogram CSVs and figures to disk.
"""
import argparse
import glob
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

RADIUS_ARRAYS = ("mis_radius", "ce_radius", "voreen_radius")


def _read_cell_radii(vtp_path: str) -> Dict[str, np.ndarray]:

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    polydata = reader.GetOutput()
    cell_data = polydata.GetCellData()
    out: Dict[str, np.ndarray] = {}
    for name in RADIUS_ARRAYS:
        arr = cell_data.GetArray(name)
        if arr is None:
            continue
        out[name] = vtk_to_numpy(arr).reshape(-1).astype(np.float64)
    return out


def _summary(values: np.ndarray) -> Dict[str, float]:

    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _plot_histogram(
    values: np.ndarray,
    name: str,
    bin_edges: np.ndarray,
    out_path: str,
):

    stats = _summary(values)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=bin_edges, color="#4c72b0", edgecolor="white")
    ax.axvline(stats["mean"], color="#c44e52", linestyle="--",
               label=f"mean={stats['mean']:.2f} mm")
    ax.axvline(stats["p50"], color="#55a868", linestyle="--",
               label=f"median={stats['p50']:.2f} mm")
    ax.set_xlabel("radius (mm)")
    ax.set_ylabel("count (cells)")
    ax.set_title(
        f"CoW centerline radius distribution — {name}\n"
        f"n={stats['count']}, p05={stats['p05']:.2f}, "
        f"p95={stats['p95']:.2f}, max={stats['max']:.2f} mm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_combined(aggregated: Dict[str, np.ndarray], out_path: str):

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, values in aggregated.items():
        if values.size == 0:
            continue
        ax.hist(
            values, bins=np.linspace(0.0, 6.0, 61),
            histtype="step", linewidth=1.6,
            label=f"{name} (n={values.size})")
    ax.set_xlabel("radius (mm)")
    ax.set_ylabel("count (cells)")
    ax.set_title("CoW centerline radius distribution — all radius arrays")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vtp_root",
        default="/home/lyyu/data/202605_topovst_rev/cow_graphs")
    parser.add_argument(
        "--out_dir",
        default="/home/lyyu/data/202605_topovst_rev/data_stats/cow_radii")
    parser.add_argument(
        "--case_prefix", default="topcow_ct_",
        help="Only include VTPs whose basename starts with this prefix.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    vtp_paths = sorted(glob.glob(os.path.join(args.vtp_root, "*.vtp")))
    vtp_paths = [
        p for p in vtp_paths
        if os.path.basename(p).startswith(args.case_prefix)]
    print(f"Found {len(vtp_paths)} VTP files with prefix '{args.case_prefix}'.")

    per_case_rows: List[Dict] = []
    chunks_by_name: Dict[str, List[np.ndarray]] = {
        n: [] for n in RADIUS_ARRAYS}

    for p in vtp_paths:
        case = os.path.basename(p)[:-len(".vtp")]
        radii_by_name = _read_cell_radii(p)
        row = {"case": case}
        for name, values in radii_by_name.items():
            chunks_by_name[name].append(values)
            stats = _summary(values)
            for k, v in stats.items():
                row[f"{name}__{k}"] = v
        per_case_rows.append(row)

    per_case_df = pd.DataFrame(per_case_rows)
    per_case_csv = os.path.join(args.out_dir, "cow_radius_per_case.csv")
    per_case_df.to_csv(per_case_csv, index=False)
    print(f"Per-case stats written to: {per_case_csv}")

    aggregated: Dict[str, np.ndarray] = {}
    overall_rows = []
    hist_edges = np.linspace(0.0, 6.0, 61)
    for name, chunks in chunks_by_name.items():
        if not chunks:
            aggregated[name] = np.array([])
            continue
        values = np.concatenate(chunks, axis=0)
        aggregated[name] = values
        stats = _summary(values)
        stats["radius_array"] = name
        overall_rows.append(stats)

        hist, _ = np.histogram(values, bins=hist_edges)
        hist_df = pd.DataFrame({
            "bin_left": hist_edges[:-1],
            "bin_right": hist_edges[1:],
            "count": hist,
        })
        hist_csv = os.path.join(args.out_dir, f"cow_radius_hist_{name}.csv")
        hist_df.to_csv(hist_csv, index=False)
        print(f"Histogram CSV for {name} written to: {hist_csv}")

        fig_path = os.path.join(args.out_dir, f"cow_radius_hist_{name}.png")
        _plot_histogram(values, name, hist_edges, fig_path)
        print(f"Histogram figure for {name} written to: {fig_path}")

    overall_df = pd.DataFrame(overall_rows)
    overall_csv = os.path.join(args.out_dir, "cow_radius_overall.csv")
    overall_df.to_csv(overall_csv, index=False)
    print(f"Overall stats written to: {overall_csv}")

    combined_path = os.path.join(args.out_dir, "cow_radius_hist_combined.png")
    _plot_combined(aggregated, combined_path)
    print(f"Combined figure written to: {combined_path}")

    print("\nOverall summary:")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
