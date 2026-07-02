# Scripts that generate training samples for different dataset.
import os
from typing import List
from multiprocessing import Pool

import pandas as pd

from src.utils.sample_transforms import (
    SampleCLModelMultiDirection,
    SampleCLModelBiDirection,
    SampleCLModelMultiDirectionRadiusAware,
    SampleCLGraphMultiDirectionRadiusAware,
    SampleSkeletonMultiDirectionRadiusAware,
    SampleSkeletonBiDirection,
)


def generate_ASOCA_samples(
    root_dir: str,
    case_names: List[str],
    save_path: str,
    in_lumen: bool = True,
    on_centerline: bool = True,
):

    dataset_name = "ASOCA"
    sample_transform = SampleCLModelMultiDirectionRadiusAware()
    sample_list = []

    case_dict = {}
    for case_name in case_names:
        case_type, _ = case_name.split("_")
        case_dict.update({
            case_name: {
                "image": os.path.join(root_dir, case_type, "CTCA",
                                      "%s.nrrd" % case_name),
                "mask": os.path.join(root_dir, case_type, "Annotations",
                                     "%s.nrrd" % case_name),
                "centerline": os.path.join(root_dir, case_type, "Centerlines",
                                           "%s.vtp" % case_name),
            },
        })

    for case_name, case_files in case_dict.items():

        data = {
            "dataset": dataset_name,
            "case_name": case_name,
            "in_lumen": in_lumen,
            "on_centerline": on_centerline,
        }
        data.update(case_files)
        sample_list.extend(sample_transform(data))

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    dataframe = pd.DataFrame(sample_list)
    dataframe.to_csv(save_path, sep=",", index=False)


def generate_Aorta24_samples(
    root_dir: str,
    case_names: List[str],
    save_path: str,
    in_lumen: bool = True,
    on_centerline: bool = True,
    phase: str = "train",
):

    dataset_name = "Aorta24"
    sample_transform = SampleCLModelMultiDirectionRadiusAware()
    sample_list = []

    if phase in ["train", "val"]:
        images_dir = "imagesTr"
        labels_dir = "labelsTr_bin"
    elif phase == "test":
        images_dir = "imagesTs"
        labels_dir = "labelsTs_bin"

    case_dict = {}
    for case_name in case_names:
        case_dict.update({
            case_name: {
                "image": os.path.join(root_dir, images_dir,
                                      "%s_0000.nii.gz" % case_name),
                "mask": os.path.join(root_dir, labels_dir,
                                     "%s.nii.gz" % case_name),
                "centerline": os.path.join(root_dir, "centerlines", "slicervmtk",
                                           "%s.vtk" % case_name),
            },
        })

    for case_name, case_files in case_dict.items():

        data = {
            "dataset": dataset_name,
            "case_name": case_name,
            "in_lumen": in_lumen,
            "on_centerline": on_centerline,
        }
        data.update(case_files)
        sample_list.extend(sample_transform(data))

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    dataframe = pd.DataFrame(sample_list)
    dataframe.to_csv(save_path, sep=",", index=False)


def generate_CoW_samples(
    nnunet_root: str,
    vtp_root: str,
    case_names: List[str],
    save_path: str,
    in_lumen: bool = True,
    on_centerline: bool = True,
    phase: str = "train",
    radius_array: str = "mis_radius",
):

    dataset_name = "CoW"
    sample_transform = SampleCLGraphMultiDirectionRadiusAware(
        radius_array=radius_array)
    sample_list = []

    if phase in ["train", "val"]:
        images_dir = "imagesTr"
        labels_dir = "labelsTr"
    elif phase == "test":
        images_dir = "imagesTs"
        labels_dir = "labelsTs"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    case_dict = {}
    for case_name in case_names:
        case_dict[case_name] = {
            "image": os.path.join(nnunet_root, images_dir,
                                  f"{case_name}_0000.nii.gz"),
            "mask": os.path.join(nnunet_root, labels_dir,
                                 f"{case_name}.nii.gz"),
            "centerline": os.path.join(vtp_root, f"{case_name}.vtp"),
        }

    for case_name, case_files in case_dict.items():
        data = {
            "dataset": dataset_name,
            "case_name": case_name,
            "in_lumen": in_lumen,
            "on_centerline": on_centerline,
        }
        data.update(case_files)
        sample_list.extend(sample_transform(data))

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    dataframe = pd.DataFrame(sample_list)
    dataframe.to_csv(save_path, sep=",", index=False)


def generate_Parse_samples(
    nnunet_root: str,
    vtp_root: str,
    case_names: List[str],
    save_path: str,
    in_lumen: bool = True,
    on_centerline: bool = True,
    phase: str = "train",
    radius_array: str = "mis_radius",
):

    dataset_name = "Parse"
    sample_transform = SampleCLGraphMultiDirectionRadiusAware(
        radius_array=radius_array)
    sample_list = []

    if phase in ["train", "val"]:
        images_dir = "imagesTr"
        labels_dir = "labelsTr"
    elif phase == "test":
        images_dir = "imagesTs"
        labels_dir = "labelsTs"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    case_dict = {}
    for case_name in case_names:
        case_dict[case_name] = {
            "image": os.path.join(nnunet_root, images_dir,
                                  f"{case_name}_0000.nii.gz"),
            "mask": os.path.join(nnunet_root, labels_dir,
                                 f"{case_name}.nii.gz"),
            "centerline": os.path.join(vtp_root, f"{case_name}.vtp"),
        }

    for case_name, case_files in case_dict.items():
        data = {
            "dataset": dataset_name,
            "case_name": case_name,
            "in_lumen": in_lumen,
            "on_centerline": on_centerline,
        }
        data.update(case_files)
        sample_list.extend(sample_transform(data))

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    dataframe = pd.DataFrame(sample_list)
    dataframe.to_csv(save_path, sep=",", index=False)


def _list_nnunet_cases(labels_dir: str):
    if not os.path.isdir(labels_dir):
        return []
    cases = []
    for fn in sorted(os.listdir(labels_dir)):
        if fn.endswith(".nii.gz"):
            cases.append(fn[:-len(".nii.gz")])
    return cases


if __name__ == "__main__":

    import argparse

    import torch

    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="CoW",
        choices=["ASOCA", "Aorta24", "CoW", "Parse"])
    parser.add_argument(
        "--cow_nnunet_root",
        default="")
    parser.add_argument(
        "--cow_vtp_root",
        default="")
    parser.add_argument(
        "--cow_out_dir",
        default="")
    parser.add_argument(
        "--cow_radius_array", default="mis_radius",
        choices=["mis_radius", "ce_radius", "voreen_radius"])
    parser.add_argument(
        "--parse_nnunet_root",
        default="")
    parser.add_argument(
        "--parse_vtp_root",
        default="")
    parser.add_argument(
        "--parse_out_dir",
        default="")
    parser.add_argument(
        "--parse_radius_array", default="mis_radius",
        choices=["mis_radius", "ce_radius", "voreen_radius"])
    args = parser.parse_args()

    if args.dataset not in ["CoW", "Parse"]:
        raise NotImplementedError(
            f"Driver for {args.dataset} is not configured in this script.")

    if args.dataset == "CoW":
        dataset_name = "CoW"
        nnunet_root = args.cow_nnunet_root
        vtp_root = args.cow_vtp_root
        out_dir = args.cow_out_dir
        radius_array = args.cow_radius_array
        generate_fn = generate_CoW_samples
    elif args.dataset == "Parse":
        dataset_name = "Parse"
        nnunet_root = args.parse_nnunet_root
        vtp_root = args.parse_vtp_root
        out_dir = args.parse_out_dir
        radius_array = args.parse_radius_array
        generate_fn = generate_Parse_samples

    train_cases = _list_nnunet_cases(os.path.join(nnunet_root, "labelsTr"))
    test_cases = _list_nnunet_cases(os.path.join(nnunet_root, "labelsTs"))
    print(f"{dataset_name} cases: train={len(train_cases)}, test={len(test_cases)}")

    splits = [("train", train_cases), ("test", test_cases)]
    for phase, cases in splits:
        if not cases:
            continue
        for kind, in_lumen, on_centerline in [
            ("on_centerline_raware_multidir", True, True),
            ("off_centerline_raware_multidir", True, False),
            ("out_lumen", False, False),
        ]:
            save_path = os.path.join(
                out_dir, f"{dataset_name}_{phase}_{kind}.csv")
            print(f"=> {save_path}")
            generate_fn(
                nnunet_root=nnunet_root,
                vtp_root=vtp_root,
                case_names=cases,
                save_path=save_path,
                in_lumen=in_lumen,
                on_centerline=on_centerline,
                phase=phase,
                radius_array=radius_array,
            )
