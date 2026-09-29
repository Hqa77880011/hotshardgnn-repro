"""Placement-independent, strictly causal temporal sampling."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleTree:
    nodes: np.ndarray
    times: np.ndarray
    children: "SampleTree | None" = None
    edge_ids: np.ndarray | None = None
    deltas: np.ndarray | None = None
    valid: np.ndarray | None = None


class TemporalSampler:
    def __init__(self, data):
        self.data = data
        node = np.r_[data.src, data.dst]
        neighbor = np.r_[data.dst, data.src]
        edge = np.tile(np.arange(len(data.src)), 2)
        order = np.lexsort((edge, data.times[edge], node))
        self.neighbor, self.edge = neighbor[order], edge[order]
        self.time = np.asarray(data.times[self.edge])
        self.ptr = np.r_[0, np.cumsum(np.bincount(node, minlength=data.num_nodes))]

    def sample(self, nodes, times, fanouts):
        """Most recent k neighbors with event time strictly below the cutoff."""
        nodes, times = np.asarray(nodes, dtype=np.int64), np.asarray(times, dtype=np.float64)
        tree = SampleTree(nodes, times)
        if not fanouts:
            return tree
        k = fanouts[0]
        child_nodes = np.repeat(nodes[:, None], k + 1, axis=1)
        child_times = np.repeat(times[:, None], k + 1, axis=1)
        edges = np.zeros((len(nodes), k), dtype=np.int64)
        valid = np.zeros((len(nodes), k), dtype=bool)
        for i, (node, cutoff) in enumerate(zip(nodes, times)):
            lo, hi = self.ptr[node:node + 2]
            end = lo + np.searchsorted(self.time[lo:hi], cutoff, side="left")
            start = max(lo, end - k)
            count = end - start
            if count:
                child_nodes[i, 1:count + 1] = self.neighbor[start:end]
                child_times[i, 1:count + 1] = self.time[start:end]
                edges[i, :count] = self.edge[start:end]
                valid[i, :count] = True
        tree.edge_ids, tree.valid = edges, valid
        tree.deltas = (times[:, None] - child_times[:, 1:]) / self.data.meta["time_scale"]
        tree.children = self.sample(child_nodes.reshape(-1), child_times.reshape(-1), fanouts[1:])
        return tree


def leaf_nodes(tree):
    while tree.children is not None:
        tree = tree.children
    return tree.nodes
