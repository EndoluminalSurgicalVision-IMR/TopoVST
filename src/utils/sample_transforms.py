import copy
import random
from typing import Dict, Any, List

import torch
import torch.multiprocessing as torchMP
import numpy as np
import networkx as nx
from tqdm import tqdm
import cupy as cp
from cucim.skimage.measure import label
from monai.transforms import Transform, Compose
from sklearn.metrics.pairwise import haversine_distances

from src.utils.geometry import transform_points, nearestPointsIndices3D, cart2spher
from src.utils.load_transforms import (
    LoadCenterlineModel,
    LoadCenterlineGraph,
    LoadFilledMask,
    LoadSkeleton,
)


def cl_model_multidir_by_intersect(
    lines: List[List[int]],
    points: np.ndarray,
    pos: np.ndarray,
):
    """
    Get centerline directions at given position. The centerline is model-based.
    """

    R = 2.0  # Sphere radius used for intersection. Unit: mm
    P_ANG = np.pi / 6  # 30 degree proximity angle
    candidates = []
    # For each line, find points lying inside sphere.
    for line in lines:
        line_pts = points[line, :]
        dist = np.linalg.norm(pos - line_pts, axis=-1, keepdims=True)
        pts_ind = np.argwhere(dist.squeeze(axis=-1) <= R)
        if len(pts_ind) == 0:
            continue
        # NOTE: Prerequisite is that the line is ordered.
        points_ind = [line[ind] for ind in pts_ind.squeeze(-1).tolist()]
        distal_points = points[[min(points_ind), max(points_ind)], :]
        directions: np.ndarray = distal_points - pos
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        if np.isnan(directions).any():
            continue
        candidates.append(directions)
    if not candidates:
        return []
    # Merge directions that are too close to each other
    candidates = np.concatenate(candidates, axis=0)  # (N, 3)
    spherecoords = cart2spher(candidates)[:, 1:] - np.array([np.pi / 2, 0])
    p_angles = np.tril(haversine_distances(spherecoords, spherecoords), k=0)
    proximity_pairs = np.argwhere((p_angles <= P_ANG) & (p_angles > 0))
    p_graph = nx.Graph()
    p_graph.add_nodes_from(list(range(len(candidates))))
    p_graph.add_edges_from(proximity_pairs.tolist())
    p_inds = list(nx.connected_components(p_graph))
    merge_directions = [
        np.mean(candidates[list(inds), :], axis=0).tolist() for inds in p_inds]

    return merge_directions


def cl_model_multidir_by_radius_aware_intersect(
    lines: List[List[int]],
    points: np.ndarray,
    radius: float,
    pos: np.ndarray,
):
    """
    Generate centerline directions at given position using radius-ball interse-
    ction. The centerline is model-based.
    """

    P_ANG = np.pi / 6  # 30 degree proximity angle
    candidates = []
    # For each line, find points lying inside radius-sphere
    for line in lines:
        line_pts = points[line, :]
        dist = np.linalg.norm(pos - line_pts, axis=-1, keepdims=True)
        pts_ind = np.argwhere(dist.squeeze(axis=-1) <= radius)
        if len(pts_ind) == 0:  # No points within sphere found
            continue
        # NOTE: Prerequisite is that the line is ordered.
        points_ind = [line[ind] for ind in pts_ind.squeeze(-1).tolist()]
        distal_points = points[[min(points_ind), max(points_ind)], :]
        directions: np.ndarray = distal_points - pos
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        if np.isnan(directions).any():
            continue
        candidates.append(directions)
    if not candidates:
        return []
    # Merge directions that are too close to each other
    candidates = np.concatenate(candidates, axis=0)  # (N, 3)
    spherecoords = cart2spher(candidates)[:, 1:] - np.array([np.pi / 2, 0])
    p_angles = np.tril(haversine_distances(spherecoords, spherecoords), k=0)
    proximity_pairs = np.argwhere((p_angles <= P_ANG) & (p_angles > 0))
    p_graph = nx.Graph()
    p_graph.add_nodes_from(list(range(len(candidates))))
    p_graph.add_edges_from(proximity_pairs.tolist())
    p_inds = list(nx.connected_components(p_graph))
    merge_directions = [
        np.mean(candidates[list(inds), :], axis=0).tolist() for inds in p_inds]

    return merge_directions


def skeleton_multidir_by_radius_aware_intersect(
    skeleton: np.ndarray,
    points: np.ndarray,
    radius: float,
    pos: np.ndarray,
    affine: np.ndarray,
):
    """
    Generate centerline directions at given position using radius-ball interse-
    ction. The centerline is skeleton-based.
    """

    dist = np.linalg.norm(pos - points, axis=-1, keepdims=True)
    pts_ind = np.argwhere(dist.squeeze(axis=-1) <= radius)
    if len(pts_ind) <= 1:  # No points within sphere found
        return []
    removal_phy = points[pts_ind].reshape(-1, 3)
    removal_img = transform_points(removal_phy, np.linalg.inv(affine))
    removal_img = np.round(removal_img, 0).astype(np.int32)

    for coord in removal_img:
        skeleton[coord[0], coord[1], coord[2]] = 0

    # From each connected component, determine its nearest end
    candidates = []
    skeleton_cp = cp.array(skeleton, dtype=cp.int32)
    conn, num_conn = label(skeleton_cp, return_num=True, connectivity=3)
    for conn_idx in range(1, num_conn + 1):
        conn_pts_cp = cp.argwhere(conn == conn_idx)
        conn_pts = cp.asnumpy(conn_pts_cp).reshape(-1, 3)
        conn_pts = transform_points(conn_pts, affine)
        conn_dist = np.linalg.norm(pos - conn_pts, axis=-1, keepdims=True)

        conn_nearest = conn_pts[np.argmin(conn_dist.squeeze(axis=-1))]
        conn_nearest = conn_nearest.reshape(-1, 3)
        direction: np.ndarray = conn_nearest - pos
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        if np.isnan(direction).any():
            continue
        candidates.append(direction.squeeze().tolist())

    return candidates


def skeleton_bidir_by_local_intersect(
    skeleton: np.ndarray,
    points: np.ndarray,
    radius: float,
    pos: np.ndarray,
    affine: np.ndarray,
):
    """
    Generate centerline directions at given position using fixed-radius ball
    intersection. The centerline is skeleton-based.
    """

    # Find indices of positions on points within distance R to given pos
    dist: np.ndarray = np.linalg.norm(pos - points, axis=-1, keepdims=True)
    pts_ind = np.argwhere(dist.squeeze(axis=-1) <= radius)
    if len(pts_ind) <= 1:  # No points within sphere found
        return []
    # Keep the old skeleton betti-0 number.
    old_skelCP = cp.array(skeleton, dtype=cp.int32)
    _, old_connNum = label(old_skelCP, return_num=True, connectivity=3)
    # Remove these points to create breakage
    removal_phy = points[pts_ind].reshape(-1, 3)
    removal_img = transform_points(removal_phy, np.linalg.inv(affine))
    removal_img = np.round(removal_img, 0).astype(np.int32)
    for coord in removal_img:
        skeleton[coord[0], coord[1], coord[2]] = 0
    # From each connected component, determine its nearest end
    candidates = []
    dila_skelCP = cp.array(skeleton, dtype=cp.int32)
    conn, num_conn = label(dila_skelCP, return_num=True, connectivity=3)
    if (num_conn - old_connNum) != 1:  # Only one additional conn component
        return []

    for conn_idx in range(1, num_conn + 1):
        conn_pts_cp = cp.argwhere(conn == conn_idx)
        conn_pts = cp.asnumpy(conn_pts_cp).reshape(-1, 3)
        conn_pts = transform_points(conn_pts, affine)
        conn_dist = np.linalg.norm(pos - conn_pts, axis=-1, keepdims=True)

        conn_nearest = conn_pts[np.argmin(conn_dist.squeeze(axis=-1))]
        conn_nearest = conn_nearest.reshape(-1, 3)
        direction: np.ndarray = conn_nearest - pos
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        if np.isnan(direction).any():
            continue
        candidates.append(direction.squeeze().tolist())

    if len(candidates) != 2:  # Double check bi-dir generation
        return []
    return candidates


def cl_model_bidir_by_interp(
    lines: List[List[int]],
    point_id: int,
    points: np.ndarray,
):
    """
    Get centerline directions at given position.
    """

    cur_point = points[point_id].reshape(-1, 3)
    tangents = []

    for line in lines:
        try:
            line_ind = line.index(point_id)
            assert line_ind >= 4 and line_ind < len(line) - 4, "Invalid index"

            neighbor_inds = [line[line_ind - 4], line[line_ind + 4]]
            neighbor_points = np.array(
                [points[n_id] for n_id in neighbor_inds]).reshape(-1, 3)
            tangents: np.ndarray = neighbor_points - cur_point
            tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

            if np.isnan(tangents).any():
                tangents = []
                continue
            if np.dot(tangents[0], tangents[1]) >= 0:
                tangents = []
                continue

            tangents = tangents.tolist()
            return tangents

        except (ValueError, AssertionError):
            continue

    return tangents


def cl_model_bidir_by_mindist(
    lines: List[List[int]],
    points: np.ndarray,
    pos: np.ndarray,
):
    """
    Get centerline direction at given position, by finding closest point on
    reference centerline and acquire direction.
    """

    # First we locate the index of nearest point on reference centerline.
    inds, _ = nearestPointsIndices3D(
        torch.from_numpy(pos), torch.from_numpy(points))
    nearest_ind = inds[0]
    tangents = []
    for line in lines:
        try:
            line_ind = line.index(nearest_ind)
            assert line_ind >= 4 and line_ind < len(line) - 4, "Invalid index"

            neighbor_inds = [line[line_ind - 4], line[line_ind + 4]]
            neighbor_points = np.array(
                [points[n_id] for n_id in neighbor_inds]).reshape(-1, 3)
            tangents: np.ndarray = neighbor_points - pos
            tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

            if np.isnan(tangents).any():
                tangents = []
                continue

            tangents = tangents.tolist()
            return tangents

        except (ValueError, AssertionError):
            continue

    return tangents


class DatasetSpecificTransform(Transform):
    """ Monai-styled transform that perform special treatment to certain
    datasets. """

    def __init__(self):

        super().__init__()

    def CoW_centerline_correction(self, data: Dict[str, Any]):
        """ Map CoW centerline coordinates into the mask's physical frame.

        CoW VTPs are stored in RAS while the segmentation masks live in LPS,
        so the x and y axes flip sign.
        """

        points: np.ndarray = data["centerline"]["points"].copy()
        points[:, 0] *= -1
        points[:, 1] *= -1
        data["centerline"]["points"] = points
        return data

    def ASOCA_centerline_correction(self, data: Dict[str, Any]):
        """ Perform centerline point coordinates correction for ASOCA dataset.

        Note:
            ASOCA provided centerlines in .vtp format. However, the coordinates
            does not match the mask loaded by ITK-tools. A correction on the
            image origin is required. Specifically, O_x, O_y are reverted
            and O_z is unchanged in the affine transform for centerline
            coordinates.
        """

        affine_itk = data["image_meta_dict"]["affine"]
        cl_affine: np.ndarray = data["image_meta_dict"]["affine"].copy()
        cl_affine[0, 3] = -1 * cl_affine[0, 3]  # O_x to -O_x
        cl_affine[1, 3] = -1 * cl_affine[1, 3]  # O_y to -O_y

        # The transformation matrix first map centerline coordinates back to
        # image coordinates, then maps them to ITK physical coordinates
        trans_matrix = affine_itk @ np.linalg.inv(cl_affine)  # (4, 4)

        points = data["centerline"]["points"]
        points = transform_points(points, trans_matrix)
        data["centerline"].update({"points": points, })

        return data

    def __call__(self, data: Dict[str, Any]):

        dataset_name = data.get("dataset")
        if dataset_name not in ["ASOCA", "CoW", "Parse"]:
            return data
        if dataset_name == "ASOCA":
            data = self.ASOCA_centerline_correction(data)
        elif dataset_name in ["CoW", "Parse"]:
            data = self.CoW_centerline_correction(data)
        return data


class SampleTransformBase(Transform):
    """ Monai-styled transform that generate a list of samples. """

    def __init__(self):

        # Subclass-specific attributes.
        self.load_transform = None
        self.post_transform = None

    @staticmethod
    def is_inside_lumen(pos: np.ndarray, mask: np.ndarray) -> bool:
        """ Test whether a position is inside the lumen mask. """

        if torch.cuda.is_available():
            device = "cuda:1"
        else:
            device = "cpu"
        # device = "cpu"

        posT = torch.from_numpy(pos).clone().to(device)
        shp = torch.tensor(mask.T.shape).to(device)  # (W, H, D)
        maskT = torch.from_numpy(mask)[
            None, None, ...].to(device).float()  # (N, C, D, H, W)
        uni_pos = ((2 * posT / shp) - 1).view(-1, 3).float()  # (-1, 1)

        test = torch.nn.functional.grid_sample(
            maskT, uni_pos.view(1, 1, 1, uni_pos.shape[0], 3), "nearest",
            padding_mode="reflection", align_corners=True,
        ).squeeze().item()

        return test > 0

    @staticmethod
    def which_outside_lumen(positions: np.ndarray, mask: np.ndarray):
        """
        Return positions that are outside of lumen.
        """

        if torch.cuda.is_available():
            device = "cuda:1"
        else:
            device = "cpu"

        posT = torch.from_numpy(positions).clone().to(device)
        shp = torch.tensor(mask.T.shape).to(device)  # (W, H, D)
        maskT = torch.from_numpy(mask)[
            None, None, ...].to(device).float()  # (N, C, D, H, W)
        uni_pos = ((2 * posT / shp) - 1).view(-1, 3).float()  # (-1, 1)

        test = torch.nn.functional.grid_sample(
            maskT, uni_pos.view(1, 1, 1, uni_pos.shape[0], 3), "nearest",
            padding_mode="reflection", align_corners=True,
        ).squeeze()

        test = torch.logical_not(test)
        return test.cpu().numpy().astype(np.uint8)

    @staticmethod
    def which_inside_lumen(positions: np.ndarray, mask: np.ndarray):
        """
        Return positions that are outside of lumen.
        """

        if torch.cuda.is_available():
            device = "cuda:1"
        else:
            device = "cpu"

        posT = torch.from_numpy(positions).clone().to(device)
        shp = torch.tensor(mask.T.shape).to(device)  # (W, H, D)
        maskT = torch.from_numpy(mask)[
            None, None, ...].to(device).float()  # (N, C, D, H, W)
        uni_pos = ((2 * posT / shp) - 1).view(-1, 3).float()  # (-1, 1)

        test = torch.nn.functional.grid_sample(
            maskT, uni_pos.view(1, 1, 1, uni_pos.shape[0], 3), "nearest",
            padding_mode="reflection", align_corners=True,
        ).squeeze()

        return test.cpu().numpy().astype(np.uint8)

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Input data must contain the following keys:
            - dataset: a str for dataset
            - case_name: current case name
            - centerline: contain points (N, 3) array and radii (N, 1) array
            - mask: (D, H, W) shaped ndarray of segmentation mask image
            - in_lumen: Whether the samples should be inside the vessel lumen
            - on_centerline: Whether the samples should be on the provided
                centerline.
        The returned list contains samples with the following keys:
            - dataset: a str for dataset
            - case: current case ID
            - position: List representation of current sample coordinate
            - radius: Radius of current sample position. Valid when
                on_centerline is True. 0 if in_lumen is False.
        """

        sample_list = []
        data = self.load_transform(data)

        dataset_name = data.get("dataset")
        case_name = data.get("case_name")
        points: np.ndarray = data["centerline"]["points"]  # (N, 3)
        radii: np.ndarray = data["centerline"]["radii"].squeeze()  # (N, )
        points_radii = list(zip(points, radii))

        mask: np.ndarray = data["mask"]
        affine: np.ndarray = data["image_meta_dict"]["affine"]

        # Base sample Dictionary
        sample_base = {
            "dataset": dataset_name,
            "case": case_name
        }

        # Generate samples based on different requirements
        in_lumen, on_centerline = data["in_lumen"], data["on_centerline"]

        if in_lumen and on_centerline:
            for point, radius in tqdm(points_radii, desc=case_name):
                sample = copy.deepcopy(sample_base)
                sample.update({
                    "position": point.tolist(),
                    "radius": radius,
                })
                sample_list.append(sample)

        elif in_lumen and not on_centerline:
            max_tries = 3
            shifts = []
            for point, radius in tqdm(points_radii, desc=case_name):
                shifted = np.random.multivariate_normal(
                    mean=point,
                    cov=np.eye(3) * 2.5 * radius,
                    size=(max_tries, ),
                ).reshape(-1, 3)  # (max_tries, 3)
                shifts.append(shifted)
            shifts = np.concatenate(shifts, axis=0)
            img_coords = transform_points(shifts, np.linalg.inv(affine))

            # Pick out inside-lumen samples from these testing points
            indicator = self.which_inside_lumen(img_coords, mask)
            print(f"{indicator.sum()} inside-lumen samples.")
            valid_shifts = shifts[indicator.nonzero()[0]].copy()
            for valid_shift in valid_shifts:
                min_inds, min_dists = nearestPointsIndices3D(
                    torch.from_numpy(valid_shift).view(1, -1), torch.from_numpy(points))
                sample = copy.deepcopy(sample_base)
                sample.update({
                    "position": valid_shift.squeeze().tolist(),
                    "radius": radii[min_inds[0]],
                })
                sample_list.append(sample)

        elif not in_lumen:
            max_tries = 3
            shifts = []
            # First we generate enough testing points
            for point, radius in tqdm(points_radii, desc=case_name):
                shifted = np.random.multivariate_normal(
                    mean=point,
                    cov=np.eye(3) * 2.5 * radius,
                    size=(max_tries, ),
                ).reshape(-1, 3)  # (max_tries, 3)
                shifts.append(shifted)
            shifts = np.concatenate(shifts, axis=0)
            img_coords = transform_points(shifts, np.linalg.inv(affine))

            # Then we pick out out-of-lumen points from these testing points
            indicator = self.which_outside_lumen(img_coords, mask)
            print(f"{indicator.sum()} out-lumen samples.")
            valid_shifts = shifts[indicator.nonzero()[0]].copy()
            for valid_shift in valid_shifts:
                sample = copy.deepcopy(sample_base)
                sample.update({
                    "position": valid_shift.squeeze().tolist(),
                    "radius": 0.0,
                })
                sample_list.append(sample)

        return sample_list


class SampleCLModelBiDirection(SampleTransformBase):
    """ Generate samples when centerlines are represented as models. """

    def __init__(self):

        self.load_transform = Compose([
            LoadFilledMask(),
            LoadCenterlineModel(),
            DatasetSpecificTransform(),
        ])

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:

        sample_list = super().__call__(data)

        # For each sample in the list, determine its directions
        print("Generating tangent directions ...")
        points = data["centerline"]["points"]
        lines = data["centerline"]["valid_lines"]
        valid_samples = []

        if data["in_lumen"] and data["on_centerline"]:
            for point_id, sample in enumerate(sample_list):
                directions = cl_model_bidir_by_interp(lines, point_id, points)
                if not directions:
                    continue
                sample.update({"tangents": directions, })
                valid_samples.append(sample)

            return valid_samples
        elif data["in_lumen"] and not data["on_centerline"]:
            for sample in sample_list:
                pos = np.array(sample["position"]).reshape(-1, 3)
                directions = cl_model_bidir_by_mindist(lines, points, pos)
                if not directions:
                    continue
                sample.update({"tangents": directions, })
                valid_samples.append(sample)

            return valid_samples
        else:
            return sample_list


class SampleCLModelMultiDirection(SampleTransformBase):
    """
    Generate samples when centerlines are represented as models. Use spherical
    intersection to determine directions.
    """

    def __init__(self):

        self.load_transform = Compose([
            LoadFilledMask(),
            LoadCenterlineModel(),
            DatasetSpecificTransform(),
        ])

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:

        sample_list = super().__call__(data)

        # For each sample in the list, determine its directions
        print("Generating tangent directions ...")
        points = data["centerline"]["points"]
        lines = data["centerline"]["valid_lines"]
        valid_samples = []
        for sample in tqdm(sample_list, desc="Tangents"):
            pos = np.array(sample["position"]).reshape(-1, 3)
            directions = cl_model_multidir_by_intersect(lines, points, pos)
            if not directions:
                continue
            sample.update({"tangents": directions, })
            valid_samples.append(sample)

        return valid_samples


class SampleCLModelMultiDirectionRadiusAware(SampleTransformBase):
    """
    Generate samples when centerlines are represented as models. Use radius-
    sphere to intersect with centerline and get directions.
    """

    def __init__(self):

        self.load_transform = Compose([
            LoadFilledMask(),
            LoadCenterlineModel(),
            DatasetSpecificTransform(),
        ])

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:

        print(f"Generating samples for {data['case_name']} ...")
        sample_list = super().__call__(data)

        # For each sample in the list, determine its directions
        print(f"Generating tangents for {data['case_name']} ...")
        points = data["centerline"]["points"]
        lines = data["centerline"]["valid_lines"]
        in_lumen, on_centerline = data["in_lumen"], data["on_centerline"]
        valid_samples = []

        # for sample in tqdm(sample_list, desc="Tangents"):
        for sample in sample_list:
            pos = np.array(sample["position"]).reshape(-1, 3)
            radius = sample["radius"]
            if radius <= 0.0 and not in_lumen and not on_centerline:
                pass  # Generating out-of-lumen samples
            elif radius <= 0.0:
                continue  # Invalid samples under other conditions
            else:  # Update tangents to multi-directional
                directions = cl_model_multidir_by_radius_aware_intersect(
                    lines, points, radius, pos)
                if not directions:
                    continue
                sample.update({"tangents": directions, })

            valid_samples.append(sample)

        print(f"{len(valid_samples)} samples selected for {data['case_name']}")
        return valid_samples


class SampleCLGraphMultiDirectionRadiusAware(SampleCLModelMultiDirectionRadiusAware):
    """ Variant of SampleCLModelMultiDirectionRadiusAware that consumes
    graph-style centerline VTPs (two-point line cells, per-cell radius).
    Used for the CoW dataset. """

    def __init__(self, radius_array: str = "mis_radius"):

        self.load_transform = Compose([
            LoadFilledMask(),
            LoadCenterlineGraph(radius_array=radius_array),
            DatasetSpecificTransform(),
        ])


class SampleSkeletonMultiDirectionRadiusAware(SampleTransformBase):
    """
    Generate samples when centerlines are represented as skeletons. Use radius
    sphere to intersect with centerline and get directions.
    """

    def __init__(self):

        self.load_transform = Compose([
            LoadFilledMask(),
            LoadSkeleton(),
            DatasetSpecificTransform(),
        ])

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:

        print(f"Generating samples for {data['case_name']} ...")
        sample_list = super().__call__(data)

        # For each sample in the list, determine its directions
        print(f"Generating tangents for {data['case_name']} ...")
        points = data["centerline"]["points"]
        affine = data["image_meta_dict"]["affine"]
        in_lumen, on_centerline = data["in_lumen"], data["on_centerline"]
        valid_samples = []

        # For efficiency, we only pick out a small portion of samples
        random.shuffle(sample_list)
        sample_list = sample_list[:250]
        for sample in tqdm(sample_list, desc="tangents"):
            skeleton = data["skeleton"].T.copy()  # (W, H, D)
            pos = np.array(sample["position"]).reshape(-1, 3)
            radius = sample["radius"]
            if radius <= 0.0 and not in_lumen and not on_centerline:
                pass  # Generating out-of-lumen samples
            elif radius <= 0.0:
                continue  # Invalid samples under other conditions
            else:
                directions = skeleton_multidir_by_radius_aware_intersect(
                    skeleton, points, radius, pos, affine)
                if not directions:
                    continue
                sample.update({"tangents": directions, })

            valid_samples.append(sample)

        print(f"{len(valid_samples)} samples selected for {data['case_name']}")
        return valid_samples


class SampleSkeletonBiDirection(SampleTransformBase):
    """
    Generate Bi-directional samples when centerline are represented as
    skeletons. Use Fix-radius sphere to intersect with skeleton and get
    directions.
    """

    def __init__(self):

        self.load_transform = Compose([
            LoadFilledMask(),
            LoadSkeleton(),
            DatasetSpecificTransform(),
        ])

    def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:

        print(f"Generating samples for {data['case_name']} ...")
        sample_list = super().__call__(data)

        # For each sample in the list, determine its directions
        print(f"Generating tangents for {data['case_name']} ...")
        points = data["centerline"]["points"]
        affine = data["image_meta_dict"]["affine"]
        skeleton: np.ndarray = data["skeleton"].T
        in_lumen, on_centerline = data["in_lumen"], data["on_centerline"]
        valid_samples = []

        # For efficiency, we only pick out a small portion of samples
        random.shuffle(sample_list)
        sample_list = sample_list[:250]
        for sample in tqdm(sample_list, desc="tangents"):
            skel = skeleton.copy()  # (W, H, D)
            pos = np.array(sample["position"]).reshape(-1, 3)
            if not in_lumen and not on_centerline:
                valid_samples.append(sample)  # Generating out-of-lumen samples
                continue
            elif in_lumen and not on_centerline:
                radius = 4.0  # off-centerline samples
            else:
                radius = 2.0  # on-centerline samples

            directions = skeleton_bidir_by_local_intersect(
                skel, points, radius, pos, affine)
            if not directions:
                continue
            sample.update({"tangents": directions, })
            valid_samples.append(sample)

        print(f"{len(valid_samples)} samples selected for {data['case_name']}")
        return valid_samples
