import copy
from typing import List

import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

from src.utils.geometry import IcoSphere, cart2spher


class BCELossGradMask(nn.Module):
    """
    BCE Loss with gradient masking.
    """

    def __init__(self, **kwargs):

        super(BCELossGradMask, self).__init__()
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwds):
        """
        Input and target should both be of shape (B, 1)
        """

        # Normalize input
        input = torch.nn.functional.sigmoid(input).view(-1, 1)
        target = target.view(-1, 1)

        assert input.shape[0] == target.shape[0], "Mismatched batch size."
        bce_loss: torch.Tensor = self.bce_loss(input, target)
        _ = bce_loss.register_hook(lambda grad: grad * target)  # Mask gradient

        return torch.sum(bce_loss)


class BCEGeoWeightedLoss(nn.Module):
    """
    BCE Loss with geometrical weighting.
    """

    def __init__(self, subdivisions: int = 3, pos_weight: float = 1.0, **kwds):

        super(BCEGeoWeightedLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction="none")
        self.sphere = IcoSphere(subdivisions=subdivisions)
        self.baseverts = self.sphere.sphereverts[:, 1:] - np.array(
            [np.pi / 2, 0])
        self.pos_w = pos_weight

    def get_weight(self, target: torch.Tensor):
        """
        Compute weight based on both class type and geometry.
        """

        with torch.no_grad():
            if torch.sum(target) == 0:
                return torch.ones_like(target).to(target.device)
            inds = torch.nonzero(target.squeeze()).squeeze().cpu().numpy()
            inpVerts = np.repeat(
                self.baseverts, len(target) // len(self.baseverts), axis=0)
            tarVerts = inpVerts[inds, :].reshape(-1, 2)
            dists = torch.from_numpy(haversine_distances(inpVerts, tarVerts))
            dists = dists.to(target.device)
            dists, _ = torch.min(dists, dim=-1, keepdim=False)
            dists: torch.Tensor = torch.nn.functional.tanh(dists)
            weights = target * self.pos_w + dists

        return weights

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwds):
        """
        Input and target should both be of shape (B, 1)
        """

        # Normalize input
        input = torch.nn.functional.sigmoid(input).view(-1, 1)
        target = target.view(-1, 1)

        assert input.shape[0] == target.shape[0], "Mismatched batch size."
        # Compute weights
        w = self.get_weight(target=target)
        base_bce = self.bce_loss(input, target)
        w_bce = base_bce * w

        return torch.mean(w_bce)


class BCEWeightedLoss(nn.Module):

    def __init__(self, **kwargs):

        super(BCEWeightedLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction="none")

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwds):
        """
        Input and target should both be of shape (B, 1)
        """

        # Normalize input
        input = torch.nn.functional.sigmoid(input).view(-1, 1)
        target = target.view(-1, 1)

        assert input.shape[0] == target.shape[0], "Mismatched batch size."
        # Use weighted BCE Loss
        with torch.no_grad():
            num_neg = torch.sum(1 - target)
            num_pos = torch.sum(target)
            if num_pos <= 0:
                w_pos = 1.0
            else:
                w_pos = (num_pos + num_neg) / num_pos  # Reciprocal
            w_neg = (num_pos + num_neg) / num_neg  # Reciprocal
            w = target * w_pos + (1 - target) * w_neg
        # Computed weighted-average BCE loss
        base_bce = self.bce_loss(input, target)
        w_bce = base_bce * w

        return torch.mean(w_bce)


class BCEFixedWeightedLoss(nn.Module):
    """
    BCE Loss with fixed weight for both pos/neg classes. Serve as baseline
    compared to other losses.
    """

    def __init__(self, pos_weight: float = 1.0, neg_weight: float = 1.0, **kwds):

        super(BCEFixedWeightedLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction="none")
        self.pos_w = pos_weight
        self.neg_w = neg_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor, **kwds):
        """
        Input and target should both be of shape (B, 1)
        """

        # Normalize input
        input = torch.nn.functional.sigmoid(input).view(-1, 1)
        target = target.view(-1, 1)

        assert input.shape[0] == target.shape[0], "Mismatched batch size."
        # Use weighted BCE Loss
        with torch.no_grad():
            w = target * self.pos_w + (1 - target) * self.neg_w
        # Computed weighted-average BCE loss
        base_bce = self.bce_loss(input, target)
        w_bce = base_bce * w

        return torch.mean(w_bce)


class BCEFocalWeightedCompoundLoss(nn.Module):

    def __init__(self, turning_step: int, subdivisions: int = 3):

        super(BCEFocalWeightedCompoundLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction="none")
        self.turning_step = turning_step
        self.sphere = IcoSphere(subdivisions=subdivisions)
        self.graph: nx.Graph = self.sphere.sphere.vertex_adjacency_graph

    def build_graph(self, response: List[float]) -> nx.Graph:
        """ Build response graph on the sphere. """

        ugraph = copy.deepcopy(self.graph)
        for n in ugraph.nodes:
            ugraph.nodes[n]["val"] = response[n]  # n is of type int

        return ugraph

    def dynamic_sample_number_weight(self, target: torch.Tensor):
        """
        Compute dynamic loss weights for given batch of target.
        """

        with torch.no_grad():
            num_neg = torch.sum(1 - target)
            num_pos = torch.sum(target)
            w_pos = (num_pos + num_neg) / num_pos  # Reciprocal
            w_neg = (num_pos + num_neg) / num_neg  # Reciprocal
            weight = target * w_pos + (1 - target) * w_neg

        return weight

    def dynamic_focal_binary_response_weight(self, input: torch.Tensor):
        """
        Compute dynamic loss weights for given input.
        """

        with torch.no_grad():
            bin_response = torch.where(input > 0.5, 1.0, 0.0)
            if not torch.any(bin_response > 0):
                return torch.zeros_like(input, device=input.device)
            num_focal = torch.sum(bin_response)
            num_other = torch.sum(1 - bin_response)
            w_focal = (num_focal + num_other) / num_focal
            w_other = (num_focal + num_other) / num_other
            weights = bin_response * w_focal + (1 - bin_response) * w_other

        return weights.to(input.device)

    def dynamic_focal_localmax_weight(self, input: torch.Tensor):
        """
        Compute dynamic loss weights for given input.
        """

        with torch.no_grad():
            response_graph = self.build_graph(input.squeeze().tolist())
            bin_response = torch.zeros_like(input.squeeze())
            for n in response_graph.nodes:
                s_paths = nx.single_source_shortest_path_length(
                    response_graph, source=n, cutoff=5)
                neighbors = list(s_paths.keys())
                neighbor_vals = torch.tensor([response_graph.nodes[n]["val"]
                                              for n in neighbors])
                node_val = response_graph.nodes[n]["val"]
                if torch.all(neighbor_vals <= node_val):
                    bin_response[n] = 1.0
                else:
                    bin_response[n] = 0.0
            num_focal = torch.sum(bin_response)
            num_other = torch.sum(1 - bin_response)
            w_focal = (num_focal + num_other) / num_focal
            w_other = (num_focal + num_other) / num_other
            weight = bin_response * w_focal + (1 - bin_response) * w_other

        return weight.to(input.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor, step: int):
        """
        Input and target should both be of shape (B, 1)
        """

        # Normalize input
        input = torch.nn.functional.sigmoid(input).view(-1, 1)
        target = target.view(-1, 1)

        assert input.shape[0] == target.shape[0], "Mismatched batch size."
        if step < self.turning_step:
            w = self.dynamic_sample_number_weight(target)
        else:
            w_sample = self.dynamic_sample_number_weight(target)
            w_focal = self.dynamic_focal_binary_response_weight(input)
            w = torch.maximum(w_sample, w_focal)  # Add focal weights

        base_bce = self.bce_loss(input, target)
        w_bce = base_bce * w

        return torch.mean(w_bce)
