# Perform graph pooling
from typing import List

import torch
import torch.nn as nn
import networkx as nx
from torch_scatter import scatter

from src.utils.geometry import IHPSphere


def getIHP_pool_unpool(levels: List[int]):
    """ Given a list of different IHP levels. Prepare IHP Grid Pooling and
    Unpooling layers between adjacent levels. """

    _levels = sorted(levels, reverse=True)
    pooling_layers = {}
    unpool_layers = {}

    for high_level, low_level in zip(_levels[:-1], _levels[1:]):
        large_IHP = IHPSphere(high_level)
        small_IHP = IHPSphere(low_level)
        large_graph = large_IHP.graph
        large_verts = torch.from_numpy(large_IHP.cartverts)
        small_verts = torch.from_numpy(small_IHP.cartverts)

        mapping = IHP_match_verts(large_verts, small_verts)
        index = IHP_scatter_index(large_graph, mapping)
        pooling_layers.update({
            "PoolingLevel%dTo%d" % (high_level, low_level): IHPGridPooling(index, low_level),
        })
        unpool_layers.update({
            "UnpoolingLevel%dTo%d" % (low_level, high_level): IHPGridUnPooling(index, high_level),
        })

    return nn.ModuleDict(pooling_layers), nn.ModuleDict(unpool_layers)


def IHP_match_verts(verts_a: torch.Tensor, verts_b: torch.Tensor):
    """ Match vertex points of one IHP to another IHP. Return a tensor
    indicating closest index in large IHP verts from small IHP grids. Inputs
    are position tensors.

    Args:
        verts_a(torch.Tensor): Shape [M, 3]
        verts_b(torch.Tensor): Shape [N, 3]

    Returns:
        Tensor of shape [N, 2]. The first column indicates indices in verts_b,
        the second column indicates corresponding indices in verts_a.
    """

    # NOTE: By building a pairwise distance matrix, we can find closest points.
    # Size: [N, M, 3]
    pairwise_diff = verts_b.unsqueeze(1) - verts_a.unsqueeze(0)
    pairwise_dist = torch.norm(pairwise_diff, dim=-1, p=2)  # Size: [N, M]

    verts_a_inds = torch.argmin(pairwise_dist, dim=1)
    mapping = torch.stack(
        (torch.tensor(list(range(verts_b.shape[0]))), verts_a_inds.squeeze()),
        dim=-1)  # [N, 2] nodes mapping tensor

    return mapping


def IHP_scatter_index(IHP_graph_a: nx.Graph, IHP_verts_map: torch.Tensor):
    """ Given verts mapping from IHP_b to IHP_a, return a scatter
    index tensor that maps IHP a feature tensor to IHP b feature
    tensor.

    Args:
        IHP_graph_a(nx.Graph): Graph object where node ids are verts indices.
        IHP_verts_map(torch.Tensor): verts indices correspondance from IHP b to
        IHP a.

    Returns:
        Tensor of Shape [M]. This tensor is used as index in scattering.
    """

    index = torch.zeros([len(IHP_graph_a.nodes)], dtype=torch.int64)
    for mapping in IHP_verts_map:
        IHP_b_inds, IHP_a_inds = mapping.tolist()
        IHP_a_neighboring_clus = \
            list(IHP_graph_a.neighbors(IHP_a_inds)) + [IHP_a_inds]
        index[IHP_a_neighboring_clus] = IHP_b_inds

    return index


class IHPGridPooling(nn.Module):
    """ A tricky implementation of IHP Graph Pooling Operations. """

    def __init__(self, index: torch.Tensor, out_level: int):

        super().__init__()
        self.index = index
        self.out_edgeindex = torch.from_numpy(IHPSphere(out_level).edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        self.index = self.index.to(x.device)
        self.out_edgeindex = self.out_edgeindex.to(edge_index.device)
        x = scatter(src=x, index=self.index, dim=0,
                    reduce="max")  # Max Pooling

        return x, self.out_edgeindex


class IHPGridUnPooling(nn.Module):
    """ A tricky implementation of IHP Graph Unpooling Operations. """

    def __init__(self, index: torch.Tensor, out_level: int):

        super().__init__()
        self.index = index
        self.out_edgeindex = torch.from_numpy(IHPSphere(out_level).edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):

        self.index = self.index.to(x.device)
        self.out_edgeindex = self.out_edgeindex.to(edge_index.device)
        x = x[self.index, :]  # Copy Unpooling

        return x, self.out_edgeindex
