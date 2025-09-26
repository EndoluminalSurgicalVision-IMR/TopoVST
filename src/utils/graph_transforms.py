# Transforms for Graph Data building.
import random
from typing import List, Dict, Any

import numpy as np
import torch
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge, BaseTransform, TwoHop
from torch_geometric.utils import k_hop_subgraph, coalesce, to_undirected
from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform, Compose
from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh

from src.utils.geometry import IcoSphere, IHPSphere, transform_points


class BuildRandScales(Transform):
    """
    Build random number of scales based on sample radius.
    """

    def __init__(self, config):

        super().__init__()
        self.min_num = config.min_num_scales
        self.max_num = config.max_num_scales
        self.min_R = config.min_scale  # Unit: mm
        self.max_R = config.max_scale  # Unit: mm
        self.fix_scales = config.fix_scales
        self.use_rand = config.rand_scales

    def __call__(self, data: Dict[str, Any]):

        if not self.use_rand:
            scales = self.fix_scales
            data.update({"scales": scales, })
            return data

        r: float = data.get("radius")
        num_scales = random.randint(self.min_num, self.max_num)
        if r is None:  # Out-of-lumen samples
            scales = np.random.rand(num_scales)
            scales = scales * (self.max_R - self.min_R) + self.min_R
            scales = scales.tolist()
        else:
            scales = []
            # Generate scales below radius
            below_r = np.random.rand(num_scales // 2)
            below_r = below_r * (r - self.min_R) + self.min_R
            scales.extend(below_r.tolist())
            # Generate scales above radius
            above_r = np.random.rand(num_scales - num_scales // 2)
            above_r = above_r * (self.max_R - r) + r
            scales.extend(above_r.tolist())

        scales.sort()
        data.update({"scales": scales, })
        return data


class BuildMultiScaleGraphStructure(Transform):
    """
    Build different scales of zero-centered icosphere graph data. No specific
    scales needed.
    """

    def __init__(self, config):

        super().__init__()
        self.sphere = IcoSphere(subdivisions=config.subdivisions)

        self.FaceToEdge = FaceToEdge(remove_faces=False)

    def __call__(self, data: Dict[str, Any]):

        scales: List[float] = data.get("scales")

        facestack, pos = self._structure_faces(scales)
        data["graph"] = Data(
            face=facestack,
            pos=pos,
            num_scales=len(scales),
            nverts=len(self.sphere.sphere.vertices),
        )
        data["graph"] = self.FaceToEdge(data["graph"])  # Build edge_index
        data["graph_meta_dict"] = {
            "nverts": torch.tensor(len(self.sphere.sphere.vertices)),
        }

        return data

    def _structure_faces(self, scales: List[float]):
        """ Get a stack of faces (reference of vertices) and vertices positions
        on the unit sphere for multiple scales. """

        # trimesh.Trimesh.faces, (n, 3), triangles
        faces = torch.from_numpy(self.sphere.sphere.faces.T).long()  # (3, n)
        pos = torch.from_numpy(self.sphere.sphere.vertices).float()
        facestack = torch.hstack([faces + i * pos.shape[0]
                                 for i in range(len(scales))])  # (3, n * Num scales)
        pos = torch.vstack([pos] * len(scales))  # (n * Num scales, 3)
        return facestack, pos


class FillSamplingCoords(Transform):
    """
    Fill in specific scale coordinates based on given scales in mm. These
    coords will be used to perform image sampling.
    """

    def __init__(self, config):

        super().__init__()

        self.sphere = IcoSphere(subdivisions=config.subdivisions)
        self.npoints = config.npoints

    def __call__(self, data: Dict[str, Any]):

        # Note the coordinate system!
        affine = data["image_meta_dict"]["affine"]
        scales: List[float] = data.get("scales")
        scales_coords = []

        # Get zero-centered ray coordinates in WORLD-coordinates
        for scale in scales:
            # Assume x-spacing is max. 0.1 cm, otherwise affine is in milimeters.
            if np.abs(affine[0, 0]) < 0.1:
                scales_coords.append(torch.from_numpy(
                    self.sphere.get_rays(self.npoints, scale / 10)).float())

            # Assume affine is in milimeters
            else:
                scales_coords.append(torch.from_numpy(
                    self.sphere.get_rays(self.npoints, scale)).float())

        # Get [Num scales * Num coords per scale, 3] array
        scales_coords = torch.cat(scales_coords, dim=0)

        origin = affine[:-1, -1]
        scales_coords += origin

        scales_coords = transform_points(
            scales_coords, torch.linalg.inv(affine))  # WORLD to Image coordinates!
        data["graph"].coords = scales_coords
        data["graph_meta_dict"].update({
            "scales": torch.tensor(scales),
        })

        return data


class BuildForGEMGCN(MapTransform):

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):

        super().__init__(keys, allow_missing_keys=allow_missing_keys)

        self.transform = Compose([
            compute_normals_edges_from_mesh,
            SimpleGeometry(),
            GemPrecomp(2, 2),
        ])

    def __call__(self, data: Dict[str, Any]):

        d = dict(data)
        for key in self.key_iterator(d):
            sample = d[key]
            sample = self.transform(sample)

            d[key] = sample

        return d


class BuildCOACTSphere(Transform):
    """ Build single-scale(radius) zero-centered icosahedral
    hexagonal-pentagonal sphere graph data with image coordinates for sampling.
    """

    def __init__(self, config: Any):

        super().__init__()
        self.ray_len = config.ray_len
        self.npoints = config.npoints

        self.sphere = IHPSphere(subdivision=config.level)

    def __call__(self, data: Dict[str, Any]):

        # Note the coordinate system!
        affine = data["image_meta_dict"]["affine"]

        # NOTE: Get zero-centered ray coordinates in WORLD-coordinates
        # Assume x-spacing is max. 0.1 cm, otherwise affine is in milimeters.
        if np.abs(affine[0, 0]) < 0.1:
            coords = torch.from_numpy(
                self.sphere.get_rays(self.npoints, self.ray_len / 10)).float()
        # Assume affine is in milimeters
        else:
            coords = torch.from_numpy(
                self.sphere.get_rays(self.npoints, self.ray_len)).float()

        data["sample_sphere"] = Data(
            coords=coords,  # zero-centered world coordinates!
            edge_index=torch.from_numpy(self.sphere.edge_index),
            num_nodes=len(self.sphere.cartverts),)
        data["sphere_meta_dict"] = {
            "ray_len": torch.tensor(self.ray_len),
            "nverts": torch.tensor(len(self.sphere.cartverts)),
        }

        return data


class NeighborExpansion(MapTransform):
    """
    Add 2^n Hop edges on the fly.
    Args:
        power(int): power of base 2 for number of hops to add to edges.
    """

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        power: int = 1,
    ):

        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if not isinstance(power, int):
            raise TypeError("Power should be int.")
        if power < 0:
            raise ValueError("Power should be int >= 0.")
        self.power = power

    def __call__(self, data: Dict[str, Any]):

        base_twoHop = TwoHop()
        d = dict(data)
        for key in self.key_iterator(d):
            sample = d[key]
            if self.power == 0:
                pass
            else:
                for _ in range(self.power):
                    sample = base_twoHop(sample)

            d[key] = sample

        return d


class GetKHopUndirectedEdges(Transform):
    """
    A transform that creates K-Hop **UNDIRECTED** edges from original
    edge_index. No self-loops are added. None edge attributes processed.
    This transform is used before graph convolution layers to **refactor** the data
    edge_index attribute. The refactored edge_index shall include edges that
    represent k-hop neighborhood.
    Current implementation is very slow and requires improvement.
    Args:
        num_hops(int): The number of hops for neighborhood localization.
        k_hop_only(bool): Whether or not to keep only the k-th order neighbors.
    """

    def __init__(self, config):

        super().__init__()
        self.k = config.num_hops
        self.k_hop_only = config.k_hop_only  # Keep only the K-hop neighbors

    def get_khop_neighbors(self, node: int, edge_index: torch.Tensor):

        subset_k, _, _, _ = k_hop_subgraph(node, self.k, edge_index)
        if self.k_hop_only and self.k > 1:
            subset_k_minus_1, _, _, _ = k_hop_subgraph(
                node, self.k - 1, edge_index)
            subset = list(set(subset_k.tolist()) -
                          set(subset_k_minus_1.tolist()))
        else:
            subset = subset_k.tolist()
            subset.remove(node)

        return subset

    def __call__(self, data: Dict[str, Any]):

        edge_index = data["graph"].edge_index  # Should be undirected
        DEV = edge_index.device
        N = data["graph"].num_nodes

        khop_edge_index = []
        # Find k-hop neighbors
        for n in range(N):
            edge_pairs = []
            neighbors = self.get_khop_neighbors(n, edge_index)
            edge_pairs = [torch.tensor([n, khop_n]) for khop_n in neighbors]
            khop_edge_index.extend(edge_pairs)
        khop_edge_index = torch.stack(khop_edge_index, dim=0).to(DEV)
        khop_edge_index = khop_edge_index.transpose(1, 0).contiguous()
        khop_edge_index, _ = coalesce(khop_edge_index, None, N)
        data["graph"].edge_index = EdgeIndex(
            khop_edge_index,
            sparse_size=(N, N),
            is_undirected=True)

        return data
