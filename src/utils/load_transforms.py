import atexit
from typing import Dict, Any, Tuple
from multiprocessing.shared_memory import SharedMemory

import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

import fill_voids
import torch
import numpy as np
import networkx as nx
import SimpleITK as sitk
from skimage.morphology import skeletonize

from monai.transforms import Transform
from monai.transforms.utils import distance_transform_edt

from src.utils.geometry import get_affine, transform_points


class LoadFilledMask(Transform):
    """ Monai-styled transform for loading mask and fill holes if exist. """

    def __init__(self):

        super().__init__()

    def __call__(self, data: Dict[str, Any]):

        mask_path = data.get("mask")

        mask_itk = sitk.ReadImage(mask_path, sitk.sitkInt32)
        affine, _ = get_affine(mask_itk)
        mask_arr = sitk.GetArrayFromImage(mask_itk)
        filled_mask = fill_voids.fill(mask_arr, in_place=False)  # [D, H, W]

        data.update({
            "mask": filled_mask,
            "image_meta_dict": {
                "affine": affine.cpu().numpy(),
                "spacing": mask_itk.GetSpacing(),
            },
        })
        return data


class LoadCenterlineModel(Transform):
    """ Monai-styled transform for loading centerline model (.vtp/.vtk). """

    MINIMUM_LINE_LEN = 10  # Add this to prevent fragments of lines

    def __init__(self):

        super().__init__()

    def __call__(self, data: Dict[str, Any]):

        centerline_path: str = data.get("centerline")

        if not centerline_path.endswith((".vtk", ".vtp")):
            raise ValueError("Only .vtk and .vtp files supported.")

        if centerline_path.endswith(".vtk"):
            reader = vtk.vtkPolyDataReader()
            arr_name = "Radius"  # Slicer-VMTK generated model
        elif centerline_path.endswith(".vtp"):
            reader = vtk.vtkXMLPolyDataReader()
            arr_name = "MaximumInscribedSphereRadius"  # ASOCA
        reader.SetFileName(centerline_path)
        reader.Update()
        model: vtk.vtkPolyData = reader.GetOutput()

        # Extract vtkPoints and convert to numpy ndarray
        points: vtk.vtkPoints = model.GetPoints()
        points = np.array([
            points.GetPoint(i)
            for i in range(points.GetNumberOfPoints())]).reshape(-1, 3)
        # Extract additional information such as radius
        # The radius information, if any
        radii = vtk_to_numpy(
            model.GetPointData().GetArray(arr_name)).reshape(-1, 1)

        # Extract lines and separate them into cells(line segments)
        lines = model.GetLines()
        lines.InitTraversal()
        num_cells = lines.GetNumberOfCells()
        valid_lines, invalid_lines = [], []
        for i in range(num_cells):
            cell = vtk.vtkIdList()
            lines.GetNextCell(cell)
            point_ids = [cell.GetId(k) for k in range(cell.GetNumberOfIds())]
            if len(point_ids) >= self.MINIMUM_LINE_LEN:
                valid_lines.append(point_ids)
            else:
                invalid_lines.append(point_ids)

        data.update({
            "centerline": {
                "points": points,
                "radii": radii,
                "valid_lines": valid_lines,
                "invalid_lines": invalid_lines,
            },
        })
        return data


class LoadCenterlineGraph(Transform):
    """ Monai-styled transform for loading a centerline that is encoded as a
    graph (each VTP cell is a 2-point edge), with per-cell radius arrays.

    Used by the CoW dataset, where vessel centerlines come as `vtkPolyData`
    with 2-point line cells and three candidate radius arrays:
    ``ce_radius``, ``mis_radius``, ``voreen_radius``. Branches (paths between
    junctions and endpoints) are reconstructed so the downstream pipeline,
    which expects ordered polylines, can consume them unchanged.
    """

    MINIMUM_LINE_LEN = 10

    def __init__(self, radius_array: str = "mis_radius"):

        super().__init__()
        self.radius_array = radius_array

    def __call__(self, data: Dict[str, Any]):

        centerline_path: str = data.get("centerline")
        if not centerline_path.endswith(".vtp"):
            raise ValueError("LoadCenterlineGraph only supports .vtp files.")

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(centerline_path)
        reader.Update()
        model: vtk.vtkPolyData = reader.GetOutput()

        vtk_points: vtk.vtkPoints = model.GetPoints()
        n_pts = vtk_points.GetNumberOfPoints()
        points = np.array([
            vtk_points.GetPoint(i) for i in range(n_pts)]).reshape(-1, 3)

        cell_radii_vtk = model.GetCellData().GetArray(self.radius_array)
        if cell_radii_vtk is None:
            raise ValueError(
                f"Cell array '{self.radius_array}' not found in {centerline_path}.")
        cell_radii = vtk_to_numpy(cell_radii_vtk).reshape(-1)

        # Each cell is a 2-point edge: (u, v).
        edges = []
        lines = model.GetLines()
        lines.InitTraversal()
        num_cells = lines.GetNumberOfCells()
        for _ in range(num_cells):
            cell = vtk.vtkIdList()
            lines.GetNextCell(cell)
            if cell.GetNumberOfIds() != 2:
                raise ValueError(
                    "LoadCenterlineGraph expects 2-point line cells; got "
                    f"{cell.GetNumberOfIds()} in {centerline_path}.")
            edges.append((cell.GetId(0), cell.GetId(1)))

        # Convert per-cell radius into per-point radius (mean over incident
        # edges; endpoints inherit their single incident edge value).
        radii_sum = np.zeros((n_pts, ), dtype=np.float64)
        radii_cnt = np.zeros((n_pts, ), dtype=np.int64)
        for (u, v), r in zip(edges, cell_radii):
            radii_sum[u] += r
            radii_cnt[u] += 1
            radii_sum[v] += r
            radii_cnt[v] += 1
        radii = np.where(radii_cnt > 0, radii_sum / np.maximum(radii_cnt, 1), 0.0)
        radii = radii.reshape(-1, 1)

        # Reconstruct ordered polylines (one per branch) from the edge graph.
        graph = nx.Graph()
        graph.add_nodes_from(range(n_pts))
        graph.add_edges_from(edges)
        degree = dict(graph.degree())
        seen = set()
        valid_lines, invalid_lines = [], []

        def walk(start: int, neighbor: int):
            path = [start, neighbor]
            prev, cur = start, neighbor
            while degree[cur] == 2:
                nxt_candidates = [
                    m for m in graph.neighbors(cur) if m != prev]
                if not nxt_candidates:
                    break
                nxt = nxt_candidates[0]
                if frozenset({cur, nxt}) in seen:
                    break
                seen.add(frozenset({cur, nxt}))
                path.append(nxt)
                prev, cur = cur, nxt
            return path

        # Walk from every non-degree-2 node (branches + endpoints).
        branch_nodes = [n for n, d in degree.items() if d != 2]
        for node in branch_nodes:
            for nb in graph.neighbors(node):
                edge = frozenset({node, nb})
                if edge in seen:
                    continue
                seen.add(edge)
                path = walk(node, nb)
                target = valid_lines if len(path) >= self.MINIMUM_LINE_LEN \
                    else invalid_lines
                target.append(path)

        # Any remaining unseen edges belong to pure cycles (no branch nodes).
        for u, v in edges:
            edge = frozenset({u, v})
            if edge in seen:
                continue
            seen.add(edge)
            # Walk the cycle starting at u in the direction of v.
            path = [u, v]
            prev, cur = u, v
            while True:
                nxt_candidates = [m for m in graph.neighbors(cur) if m != prev]
                if not nxt_candidates:
                    break
                nxt = nxt_candidates[0]
                edge_n = frozenset({cur, nxt})
                if edge_n in seen:
                    break
                seen.add(edge_n)
                path.append(nxt)
                prev, cur = cur, nxt
                if cur == u:
                    break
            target = valid_lines if len(path) >= self.MINIMUM_LINE_LEN \
                else invalid_lines
            target.append(path)

        data.update({
            "centerline": {
                "points": points,
                "radii": radii,
                "valid_lines": valid_lines,
                "invalid_lines": invalid_lines,
            },
        })
        return data


class LoadSkeleton(Transform):
    """ Monai-styled transform for loading skeleton image (e.g., NifTi). """

    def __init__(self):

        super().__init__()

    def extract_radii_dt(self, data: Dict[str, Any], skeleton: np.ndarray):
        """ Extract radii from skeleton image. """

        if torch.cuda.is_available():
            DEV = "cuda"
        else:
            DEV = "cpu"

        maskT = torch.from_numpy(data.get("mask"))  # [D, H, W]
        skelT = torch.from_numpy(skeleton.T)  # [W, H, D]
        affine: torch.Tensor = torch.from_numpy(
            data["image_meta_dict"]["affine"])
        spacing: Tuple = data["image_meta_dict"]["spacing"]
        # Minimum radius should be diagonal-distance of one voxel
        min_radius = torch.linalg.norm(torch.tensor(spacing)).item()

        dtPhy = distance_transform_edt(
            maskT.unsqueeze(0).to(DEV), sampling=spacing[::-1])
        dtPhy = dtPhy.squeeze(0).float()
        coords = torch.argwhere(skelT).view(-1, 3).to(DEV)  # [Ix, Iy, Iz]
        s_coords = coords * 2 / torch.tensor(skelT.shape).to(coords.device) - 1
        s_coords = s_coords.view(-1, 3).float()

        phy_radius = torch.nn.functional.grid_sample(
            dtPhy[None, None, ...],
            s_coords.view(1, 1, 1, s_coords.shape[0], 3),
            mode="bilinear", padding_mode="reflection", align_corners=True,)
        phy_radius = phy_radius.squeeze().view(-1, 1)
        phy_radius[phy_radius < min_radius] = min_radius
        phy_coords = transform_points(coords, affine.to(DEV)).view(-1, 3)

        data.update({
            "centerline": {
                "points": phy_coords.cpu().numpy(),
                "radii": phy_radius.cpu().numpy(),
            },
        })

        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:

        # Load skeleton image and mask image.
        skeleton_path: str = data.get("centerline", None)
        mask_arr: np.ndarray = data.get("mask")

        if skeleton_path is not None:
            skel_itk = sitk.ReadImage(skeleton_path, sitk.sitkUInt8)
            skeleton = sitk.GetArrayFromImage(skel_itk)  # (D, H, W)
        else:
            skeleton = skeletonize(mask_arr).astype(np.int32)
        data.update({"mask": mask_arr, "skeleton": skeleton, })
        data = self.extract_radii_dt(data, skeleton)

        return data


class LoadImageMaskRTCached(Transform):
    """ Return a monai-styled image with segmentation mask. Use run-time cache
    to manage image loading with efficiency.
    """

    def __init__(self):

        super().__init__()
        # Memory cache for faster loading
        self.cache: Dict[str, SharedMemory] = {}

    @staticmethod
    def createSharedMemory(arr: np.ndarray, name: str):
        """
        Create a piece of shared memory object.
        """

        shm = SharedMemory(name=name, create=True, size=arr.nbytes)
        # Create a numpy array proxy to fill in arr data.
        proxy = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        proxy[:] = arr[:]

        return shm

    def cleanup(self):

        for shm_holder in self.cache.values():
            try:
                # shm_holder.close()
                shm_holder.unlink()
            except Exception:
                continue

    def __call__(self, data: Dict[str, Any]):

        image_name = data["image"]
        mask_name = data["mask"]
        case_name = data["case_name"]

        try:
            # Access Shared memory
            shpMem = self.cache[f"{case_name}_shape"]
            shape = np.ndarray(shape=(3, ), dtype=np.int32, buffer=shpMem.buf)
            imgMem = self.cache[f"{case_name}_image"]
            image = np.ndarray(shape=shape.tolist(),
                               dtype=np.float32, buffer=imgMem.buf)
            maskMem = self.cache[f"{case_name}_mask"]
            mask = np.ndarray(shape=shape.tolist(),
                              dtype=np.uint8, buffer=maskMem.buf)
            affMem = self.cache[f"{case_name}_affine"]
            affine = np.ndarray(
                shape=(4, 4), dtype=np.float32, buffer=affMem.buf)
            spcMem = self.cache[f"{case_name}_affine"]
            spacing = np.ndarray(
                shape=(3, ), dtype=np.float32, buffer=spcMem.buf)

        except (FileNotFoundError, KeyError):
            # Use SimpleITK to load image and mask
            image_sitk = sitk.ReadImage(image_name, sitk.sitkFloat32)
            mask_sitk = sitk.ReadImage(mask_name, sitk.sitkUInt8)
            affine, _ = get_affine(image_sitk)
            spacing = image_sitk.GetSpacing()

            # Convert to Numpy arrays with specified types
            shape = np.array(image_sitk.GetSize(), dtype=np.int32)
            image = sitk.GetArrayFromImage(image_sitk)  # (D, H, W)
            old_mask = sitk.GetArrayFromImage(mask_sitk)  # (D, H, W)
            mask: np.ndarray = fill_voids.fill(old_mask, in_place=False)
            mask = mask.astype(np.uint8)
            affine = affine.cpu().numpy().astype(np.float32)
            spacing = np.array(spacing, dtype=np.float32)

            # Create memory buffers
            try:
                self.cache[f"{case_name}_shape"] = self.createSharedMemory(
                    shape, f"{case_name}_shape")
                self.cache[f"{case_name}_image"] = self.createSharedMemory(
                    image, f"{case_name}_image")
                self.cache[f"{case_name}_mask"] = self.createSharedMemory(
                    mask, f"{case_name}_mask")
                self.cache[f"{case_name}_affine"] = self.createSharedMemory(
                    affine, f"{case_name}_affine")
                self.cache[f"{case_name}_spacing"] = self.createSharedMemory(
                    spacing, f"{case_name}_spacing")
            except FileExistsError:
                pass  # Race condition handler

        # Deep-copy the data when returning
        data.update({
            "image": torch.from_numpy(image.copy()),  # Same memory
            "mask": torch.from_numpy(mask.copy()),  # Same memory
            "image_meta_dict": {
                "affine": torch.from_numpy(affine.copy()),  # Same memory
                "spacing": spacing.tolist(),  # Copied memory
            },
        })
        return data


class LoadImageMask(Transform):
    """ Return a monai-styled image with segmentation mask. """

    def __init__(self):

        super().__init__()

    def __call__(self, data: Dict[str, Any]):

        image_name = data["image"]
        mask_name = data["mask"]

        # Use SimpleITK to load image and mask
        image_sitk = sitk.ReadImage(image_name, sitk.sitkFloat32)
        mask_sitk = sitk.ReadImage(mask_name, sitk.sitkUInt8)
        affine, _ = get_affine(image_sitk)

        old_mask = sitk.GetArrayFromImage(mask_sitk)  # (D, H, W)
        filled_mask = fill_voids.fill(old_mask, in_place=False)

        data.update({
            "image": torch.from_numpy(sitk.GetArrayFromImage(image_sitk)),
            "mask": torch.from_numpy(filled_mask),
            "image_meta_dict": {
                "affine": affine,
                "spacing": image_sitk.GetSpacing(),
            },
        })
        return data
