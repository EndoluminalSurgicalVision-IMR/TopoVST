# Scripts that generate training samples for different dataset.
import os
from typing import List
from multiprocessing import Pool

import pandas as pd

from src.utils.sample_transforms import (
    SampleCLModelMultiDirection,
    SampleCLModelBiDirection,
    SampleCLModelMultiDirectionRadiusAware,
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


if __name__ == "__main__":

    import torch

    torch.multiprocessing.set_start_method("spawn")

    # TODO: Prepare dataset
    dataset = "ASOCA"
    root_dir = ""
    phase = ""  # "train", "val"

    # TODO: Prepare case_names
    case_names = []

    # TODO: Generate samples with different configurations
    save_path = f"tmp_data/samples/{dataset}_{phase}_on_centerline_raware_multidir.csv"
    generate_ASOCA_samples(root_dir, case_names, save_path, True, True)
    save_path = f"tmp_data/samples/{dataset}_{phase}_off_centerline_raware_multidir.csv"
    generate_ASOCA_samples(root_dir, case_names, save_path, True, False)
    save_path = f"tmp_data/samples/{dataset}_{phase}_out_lumen.csv"
    generate_ASOCA_samples(root_dir, case_names, save_path, False, False)
