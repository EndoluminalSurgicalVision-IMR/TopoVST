import copy
import math
from typing import Dict, Any, Tuple, List

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from monai.transforms import Transform
import numpy as np
import networkx as nx

from src.utils.geometry import IcoSphere
from src.utils.image_transforms import SampleIcoSphere
from src.model.adasire.ada_sire_trackers import (
    AdaSIREMultiTaskMultiLossTracker,
    AdaSIREMultiTaskScaleFusionTracker,
)


class AdaSireSampleAndPredict(Transform):
    """
    Wrapper of AdaSIRE framework sampling and inference pipeline.
    """

    def __init__(self, config):

        self.sampler = SampleIcoSphere(config=config)
        self.device = config.device
        self.pl_module = config.pl_module

        # AdaSIRE Tracker model
        self.tracker_model = config.pl_module.load_from_checkpoint(
            config.mdl_ckpt, map_location="cpu").eval().to(config.device)

    def __call__(self, data: Dict[str, Any]):

        point: torch.Tensor = data.get("point")
        point = point.to(self.device)

        sampled_data = self.sampler(data, point.clone())
        sampled_data = list(DataLoader([sampled_data]))[0]

        save_features = data.get("save_features")
        save_weights = data.get("save_weights")

        # For different Pytorch Lightning Modules (different networks),
        # extract the corresponding return items.
        if isinstance(self.tracker_model, AdaSIREMultiTaskMultiLossTracker):
            scales_ft, radii_ft, heatmap = self.tracker_model(sampled_data)
            heatmap = heatmap.tolist()
            scales_ft = scales_ft.tolist()
            radii_ft = radii_ft.tolist()
            data.update(
                {"pred": heatmap, "scales_ft": scales_ft, "radii_ft": radii_ft, })
        elif isinstance(self.tracker_model, AdaSIREMultiTaskScaleFusionTracker):
            res: Tuple[torch.Tensor] = self.tracker_model(
                sampled_data["sample"]["graph"], save_features=save_features)
            heatmap = res[0].tolist()
            radius = res[1].item()
            radius_param = data.get("radius_param", "logit")
            if radius_param == "log":
                radius = data["r_ref"] * math.exp(radius)
            elif data.get("base_radius") is not None:
                radius = F.sigmoid(torch.tensor(radius)).item()
                radius = radius * data.get("base_radius")
            weights = res[2][0].tolist() if save_weights else None
            data.update(
                {"pred": heatmap, "radius": radius, "weights": weights, })
            if save_features:
                data.update({"features": res[3].tolist(), })
        else:
            raise NotImplementedError(f"{self.pl_module} not ready.")

        if data.get("save_sample", False):
            data.update({
                "spheres": sampled_data["sample"]["spheres"].features.tolist(),
            })
        del sampled_data
        torch.cuda.empty_cache()

        return data


class AdaSireScalesNormMax(Transform):
    """
    Perform normalization and argmax across scales for SIRE.
    """

    def __init__(self, config) -> None:

        self.device = config.device

    def __call__(self, data: Dict[str, Any]):

        # Extract logits on each scales
        scales_ft = data.get("scales_ft")
        scales_ft = torch.tensor(scales_ft).to(self.device).squeeze()
        # Perform normalization
        norm_style = data.get("normalization")
        if norm_style == "relu":
            scales_ft = F.relu(scales_ft)
        elif norm_style == "softmax":
            scales_ft = F.softmax(scales_ft, dim=-1)
        elif norm_style == "sigmoid":
            scales_ft = F.sigmoid(scales_ft)
        elif norm_style == "none":
            pass
        # Perform Argmax
        pred, _ = torch.max(scales_ft, dim=0, keepdim=False)

        data.update({
            "scales_ft": scales_ft.tolist(),
            "pred": pred.tolist(),
        })

        return data


class AdaSirePredictionNormalize(Transform):
    """
    Perform normalization on predicted sphere.
    """

    def __init__(self, config) -> None:

        self.device = config.device

    def __call__(self, data: Dict[str, Any]):

        pred = data.get("pred")
        pred = torch.tensor(pred).to(self.device).squeeze()

        # Perform normalization
        norm_style = data.get("normalization")
        if norm_style == "relu":
            pred = F.relu(pred)
        elif norm_style == "softmax":
            pred = F.softmax(pred, dim=0)
        elif norm_style == "sigmoid":
            pred = F.sigmoid(pred)
        elif norm_style == "none":
            pass

        data.update({
            "pred": pred.tolist(),
        })

        return data


class AdaSirePhyRadius(Transform):
    """
    Perform physical radius regression using predictions on different scales.
    """

    def __init__(self, config) -> None:

        self.device = config.device
        self.phy_scales = config.scales

    def __call__(self, data: Dict[str, Any]):

        radii_ft = data.get("radii_ft")
        radii = torch.tensor(radii_ft).to(self.device).squeeze()

        inds = torch.argwhere((radii > 0.2) & (radii < 0.8))
        mask = torch.zeros_like(radii).float()
        mask[inds] = 1.0

        if not torch.any(mask > 0):
            phy_r = 0.25
        else:
            scales = torch.tensor(self.phy_scales).to(self.device)
            phy_r = torch.sum(scales * radii * mask) / torch.sum(mask)
            phy_r = phy_r.item()

        data.update({
            "radius": phy_r,
        })

        return data


class IcoSphereLocalMaxSolver(Transform):
    """
    Transform that performs local-maximum extraction on a real-valued sphere.
    """

    def __init__(self, config, use_config: bool = True, **kwds) -> None:

        if use_config:
            self.sphere = IcoSphere(config.subdivisions)
            self.device = config.device
        else:
            self.sphere = IcoSphere(kwds.get("subdivisions"))
            self.device = kwds.get("device")
        self.graph = self.sphere.sphere.vertex_adjacency_graph

    def graph_localMax(self, t: torch.Tensor) -> nx.Graph:
        """
        Find local max points on the graph, return a binary graph indicator.
        """

        # Use 95%-percentile value as threshold
        # threshold = np.percentile(t.cpu().numpy(), 95)
        # Use 0.5 as threshold
        threshold = 0.5
        t[t <= threshold] = 0

        tmp_graph = self.build_graph(t.squeeze().tolist())
        bin_graph = self.build_graph(torch.zeros_like(t).squeeze().tolist())

        # For each node in the graph, find its local neighbors and determine
        # local maximum property. Search depth=2
        for node in tmp_graph.nodes:
            s_paths = nx.single_source_shortest_path_length(
                tmp_graph, source=node, cutoff=2)
            neighbors = list(s_paths.keys())
            neighbor_vals = torch.tensor([tmp_graph.nodes[n]["response"]
                                          for n in neighbors])
            node_val = tmp_graph.nodes[node]["response"]
            if torch.all(neighbor_vals == 0) and node_val == 0:
                bin_graph.nodes[node]["response"] = 0  # Special case
            elif torch.all(neighbor_vals <= node_val):
                bin_graph.nodes[node]["response"] = 1
            else:
                bin_graph.nodes[node]["response"] = 0

        return bin_graph

    def build_graph(self, response: List[float]) -> nx.Graph:
        """ Build response graph on the sphere. """

        ugraph = copy.deepcopy(self.graph)
        for n in ugraph.nodes:
            ugraph.nodes[n]["response"] = response[n]  # n is of type int
            ugraph.nodes[n]["pos"] = self.sphere.cartverts[n, :].copy()

        return ugraph

    def get_conn_subgraph(self, graph: nx.Graph):
        """ Get connected subgraphs from binary response graph. """

        return graph.subgraph([n for n, data in graph.nodes(data=True)
                               if data["response"] > 0])

    def mean_dir_Cartesian(self, cartverts: np.ndarray) -> np.ndarray:
        """ Compute mean direction on a sphere using cartesian coordinates of
        directions on the sphere. """

        mean_dir = np.mean(cartverts, axis=0).squeeze()
        mean_dir /= np.linalg.norm(mean_dir)

        return mean_dir

    def direction_from_conn_regions(self, graph: nx.Graph) -> List[List]:
        """ Get connected components of response graph and compute direction.
        """

        conn_regions = list(nx.connected_components(graph))
        # conn_regions = self.conn_region_filter(conn_regions)

        directions = []
        for nodes in conn_regions:
            pos = [graph.nodes[n]["pos"] for n in nodes]
            mean_dir = self.mean_dir_Cartesian(np.array(pos).reshape(-1, 3))
            directions.append(mean_dir.tolist())

        return directions

    def __call__(self, data: Dict[str, Any]):

        pred: List | torch.Tensor = data.get("pred")
        if torch.is_tensor(pred):
            t = pred.clone().detach().to(self.device)
        else:
            t = torch.tensor(pred).to(self.device)

        localmax = self.graph_localMax(t=t.clone())
        subgraph = self.get_conn_subgraph(localmax)
        directions = self.direction_from_conn_regions(subgraph)

        data.update({
            "bin_pred": [d["response"] for _, d in localmax.nodes(data=True)],
            "directions": directions,
        })

        return data
