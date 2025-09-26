# Tracked centerline tree structure
import copy
from typing import Dict, Any, List, Set
from collections import namedtuple

import networkx as nx
import torch
import numpy as np


# Node type that will be inserted into a queue during DFS or BFS.
ANCHOR = namedtuple("Anchor", ["node", "position", "radius"])


class TrackedCenterline(object):
    """ Maintains a tracked tree-structured centerline graph. """

    def __init__(self):

        self.cur_node = -1  # Node indicator
        self.centerline_graph = nx.Graph()

    def reset(self):
        """ Clear all tracked points and their properties. """

        self.cur_node = -1
        self.centerline_graph.clear()

    def add_point(
        self,
        position: torch.Tensor | np.ndarray | List[float],
        parent_node: int,
        **kwds,
    ) -> int:
        """ Add node to current graph and edge between new node and
        parent node. """

        self.cur_node += 1  # Assign new node indicator

        # Insert new node with given position.
        if isinstance(position, (torch.Tensor, np.ndarray)):
            self.centerline_graph.add_node(
                self.cur_node, pos=position.tolist(), **kwds)
        elif isinstance(position, List):
            self.centerline_graph.add_node(self.cur_node, pos=position, **kwds)
        else:
            raise ValueError("Input position should be Tensor, array or list.")
        # Insert new edge if parent node is specified
        if parent_node == -1:  # -1 means no parent node for current position
            pass
        elif isinstance(parent_node, int):
            self.centerline_graph.add_edge(parent_node, self.cur_node)
        else:
            raise ValueError("Input parent node should be int.")

        return self.cur_node  # Return the node added

    def add_edge(self, n1: int, n2: int):
        """
        Add an edge between given node n1, n2.
        """

        # Check node existence first, since add_edge will automatically add
        # missing nodes.
        if n1 not in self.centerline_graph.nodes:
            raise ValueError(f"{n1} not in current tracked centerline.")
        if n2 not in self.centerline_graph.nodes:
            raise ValueError(f"{n2} not in current tracked centerline.")

        self.centerline_graph.add_edge(n1, n2)

    def update_node_attr(self, node: int, **kwds,):
        """ Update node attributes in current graph. """

        for k, v in kwds.items():
            self.centerline_graph.nodes[node][k] = v

    def __len__(self):
        """ Return current length of tracked centerline as the total number
        of inserted nodes in the graph. """

        return len(self.centerline_graph.nodes)

    def post_process(self):
        """
        Add missing edges between overlapping endpoints(degree=1) and isolated
        nodes(degree=0).
        """

        new_edges = []
        graph = copy.deepcopy(self.centerline_graph)
        conns = list(nx.connected_components(graph))

        # First we add edges between different connected components
        for conn in conns:
            key_nodes = [n for n in conn if graph.degree[n] <= 1]
            search_pos, search_r, search_id = [], [], []
            for n, d in graph.nodes(data=True):
                search_id.append(n)
                search_pos.append(d["pos"])
                search_r.append(d["r"] * 1.5)
            search_pos = np.array(search_pos).reshape(-1, 3)

            for k in key_nodes:
                k_pos = np.array(graph.nodes[k]["pos"]).reshape(-1, 3)
                k_r = graph.nodes[k]["r"] * 1.5
                dists = np.linalg.norm(
                    search_pos - k_pos, ord=2, axis=-1, keepdims=False)
                thres = np.array(search_r) + k_r
                targets = [search_id[t] for t in
                           np.argwhere(dists <= thres).squeeze(-1).tolist()]
                subG: List[Set] = list(
                    nx.connected_components(graph.subgraph(targets)))

                for sub in subG:
                    if sub.issubset(conn):
                        continue
                    sub = list(sub)
                    if len(sub) == 1:
                        new_edges.append((k, sub[0]))
                    else:
                        sub_dists = dists[sub]
                        new_edges.append(
                            (k, sub[np.argmin(sub_dists, axis=0, keepdims=False)]))

        new_graph = copy.deepcopy(graph)
        for new_edge in new_edges:
            new_graph.add_edge(new_edge[0], new_edge[1])
            if len(nx.cycle_basis(new_graph, new_edge[0])) > 0:
                new_graph.remove_edge(new_edge[0], new_edge[1])
            if len(nx.cycle_basis(new_graph, new_edge[1])) > 0:
                new_graph.remove_edge(new_edge[0], new_edge[1])

        self.centerline_graph = new_graph

    @property
    def cursor(self) -> int:
        """ Return an integer indicating current centerline node. """

        return self.cur_node

    @property
    def centerline_r(self) -> torch.Tensor:
        """ Return a tensor containing current centerline radii, if exists. """

        return torch.tensor(
            [data["r"] for _, data in self.centerline_graph.nodes(data=True)])

    @property
    def centerline_points(self) -> torch.Tensor:
        """ Return a tensor containing current centerline positions. """

        return torch.tensor(
            [data["pos"] for _, data in self.centerline_graph.nodes(data=True)]).view(-1, 3)

    @property
    def centerline_graph_dict(self) -> Dict[str, Any]:
        """ Return a dict description of centerline graph. """

        graph = {
            "nodes": [],
            "edges": [],
        }

        for node, data in self.centerline_graph.nodes(data=True):
            node_entry = {
                "id": node,
                "attributes": data,
            }
            graph["nodes"].append(node_entry)

        for edge in self.centerline_graph.edges:
            graph["edges"].append(edge)

        return graph

    @property
    def centerline_graph_nx(self) -> nx.Graph:
        """ Return a networkx Graph object of centerline graph. """

        return copy.deepcopy(self.centerline_graph)

    @property
    def centerline_graph_simple_nx(self) -> nx.Graph:
        """ Return a networkx Graph object with only pos node attr. """

        save_graph = nx.Graph()
        for node, data in self.centerline_graph.nodes(data=True):
            attr = {"pos": data["pos"], }
            save_graph.add_node(node, **attr)
        save_graph.add_edges_from(self.centerline_graph.edges())
        return save_graph
