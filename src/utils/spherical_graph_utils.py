# Implements algorithms necessary for matching points on spheres.
# Required by COACT Tracker
from typing import List

import torch
from monai.transforms import Transform
import networkx as nx

from src.utils.geometry import IcoSphere


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


class LocalMaxEvaluator():
    """
    Direction Evaluator during training/validation stage.
    """

    def __init__(self, **kwds):

        self.dir_transform = IcoSphereLocalMaxSolver(None, False, **kwds)

    @staticmethod
    def cos_similarity(x1: List[List[float]], x2: List[List[float]]) -> float:
        """
        Compute cosine similarity between predicted directions and label
        directions.
        """

        x1 = torch.tensor(x1).view(-1, 3).unsqueeze(1)
        x2 = torch.tensor(x2).view(-1, 3).unsqueeze(0)

        sim_mat = torch.nn.functional.cosine_similarity(x1, x2, dim=-1)
        similarity = sim_mat.max(dim=-1, keepdim=False)[0]

        return torch.mean(similarity).item()

    def normalize(self, t: torch.Tensor):
        """
        Normalize the prediction before further evaluation
        """

        # Sigmoid normalization
        t = torch.nn.functional.sigmoid(t)
        return t

    def __call__(self, pred: torch.Tensor, label: torch.Tensor):

        pred = self.normalize(pred)

        # Process batches
        bs = pred.shape[0]

        numErrs = 0
        cosSims = 0
        for b in range(bs):
            pred_d = self.dir_transform({
                "pred": pred[b],
            })
            label_d = self.dir_transform({
                "pred": label[b],
            })
            num_err = len(pred_d["directions"]) - len(label_d["directions"])
            if not pred_d["directions"] and not label_d["directions"]:
                cossim = 1.0
            elif not pred_d["directions"] or not label_d["directions"]:
                cossim = 0.0
            else:
                cossim = self.cos_similarity(
                    pred_d["directions"], label_d["directions"])

            numErrs += num_err
            cosSims += cossim

        return {
            "num_response_err": numErrs / bs,
            "cosine_similarity": cosSims / bs,
        }
