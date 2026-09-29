"""Deterministic JODIE ingestion and small offline integration fixtures."""

import argparse
import csv
import json
from pathlib import Path
import urllib.request

import numpy as np


JODIE = {name: f"https://snap.stanford.edu/jodie/{name}.csv"
         for name in ("wikipedia", "reddit", "mooc", "lastfm")}


def save_dataset(path, arrays, metadata):
    path = Path(path)
    if (path / "metadata.json").exists():
        raise FileExistsError(f"Dataset already exists: {path}; choose another output directory")
    path.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(path / f"{name}.npy", value, allow_pickle=False)
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))
        for name in self.meta["arrays"]:
            setattr(self, name, np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False))
        self.num_nodes = len(self.features)


def prepare_jodie(name, output, raw_dir):
    """Download official CSV, remap bipartite IDs, preserve chronological order."""
    raw = Path(raw_dir) / f"{name}.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        partial = raw.with_suffix(".csv.part")
        urllib.request.urlretrieve(JODIE[name], partial)
        partial.replace(raw)
    users, items, times, features = [], [], [], []
    with raw.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for line, row in enumerate(reader, 2):
            if len(row) < 4:
                raise ValueError(f"Malformed JODIE CSV at line {line}")
            users.append(int(row[0]))
            items.append(int(row[1]))
            times.append(float(row[2]))
            features.append([float(v) for v in row[4:]])
    _, src = np.unique(users, return_inverse=True)
    item_ids, dst = np.unique(items, return_inverse=True)
    num_users = int(src.max()) + 1
    dst += num_users
    times = np.asarray(times, dtype=np.float64)
    edge_features = np.asarray(features, dtype=np.float32)
    if not np.isfinite(times).all() or not np.isfinite(edge_features).all():
        raise ValueError("Dataset contains non-finite timestamps/features")
    order = np.argsort(times, kind="stable")
    times = times[order] - times.min()
    positive_gaps = np.diff(times)
    time_scale = float(np.median(positive_gaps[positive_gaps > 0]))
    dim = max(edge_features.shape[1], 32)
    arrays = dict(src=src[order].astype(np.int64), dst=dst[order].astype(np.int64),
                  times=times, edge_features=edge_features[order],
                  features=np.zeros((num_users + len(item_ids), dim), dtype=np.float32))
    train_end, val_end = np.quantile(times, [0.70, 0.85])
    arrays["split"] = np.where(times <= train_end, 0, np.where(times <= val_end, 1, 2)).astype(np.int8)
    save_dataset(output, arrays, dict(name=name, task="link", source=JODIE[name],
                 num_users=num_users, num_items=len(item_ids), events=len(times),
                 time_scale=time_scale, arrays=list(arrays),
                 split="chronological timestamp quantiles 70/15/15; tied timestamps stay together",
                 node_features="zero-filled; raw JODIE dimensions are edge features"))


def synthetic(output, seed=7, events=4096, nodes=128, task="link"):
    """Small generated graph for correctness checks, not paper performance data."""
    if nodes < 8 or events < 32 or nodes % 2:
        raise ValueError("Synthetic fixture requires an even node count >=8 and events >=32")
    rng = np.random.default_rng(seed)
    users = nodes // 2
    src = rng.integers(users, size=events)
    dst = users + (src + rng.integers(0, 8, size=events)) % users
    arrays = dict(src=src.astype(np.int64), dst=dst.astype(np.int64),
                  times=np.arange(events, dtype=np.float64),
                  edge_features=rng.normal(size=(events, 4)).astype(np.float32),
                  features=rng.normal(size=(nodes, 16)).astype(np.float32),
                  split=np.repeat([0, 1, 2], [int(events * .7), int(events * .15),
                                             events - int(events * .7) - int(events * .15)]).astype(np.int8))
    if task == "node":
        arrays["labels"] = (arrays["features"][:, 0] > 0).astype(np.int64)
        arrays["node_split"] = np.repeat([0, 1, 2], [nodes // 2, nodes // 4,
                                                    nodes - nodes // 2 - nodes // 4]).astype(np.int8)
    save_dataset(output, arrays, dict(name="synthetic", task=task, generated=True,
                 seed=seed, num_users=users, num_items=users, events=events,
                 time_scale=1.0, arrays=list(arrays)))


def prepare_products(output, raw_dir, seed):
    """OGBN-Products with explicit, reproducible synthetic edge timestamps."""
    from ogb.nodeproppred import NodePropPredDataset

    dataset = NodePropPredDataset(name="ogbn-products", root=raw_dir)
    graph, labels = dataset[0]
    source, target = graph["edge_index"]
    # OGB contains both directions. Keep one canonical unordered pair.
    keep = source < target
    source, target = source[keep].astype(np.int64), target[keep].astype(np.int64)
    # Seeded uniform arrival order is a disclosed reconstruction choice.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(source))
    node_split = np.full(len(labels), 2, dtype=np.int8)
    indices = dataset.get_idx_split()
    node_split[indices["train"]] = 0
    node_split[indices["valid"]] = 1
    arrays = dict(src=source[order], dst=target[order],
                  times=np.arange(len(source), dtype=np.float64),
                  edge_features=np.empty((len(source), 0), dtype=np.float32),
                  features=graph["node_feat"].astype(np.float32),
                  split=np.zeros(len(source), dtype=np.int8),
                  labels=labels.reshape(-1).astype(np.int64), node_split=node_split)
    save_dataset(output, arrays, dict(name="products-t", task="node", seed=seed,
                 source="https://ogb.stanford.edu/docs/nodeprop/#ogbn-products",
                 generated_timestamps=True, timestamp_rule="seeded permutation of canonical undirected edges",
                 time_scale=1.0, arrays=list(arrays), events=len(source)))


def partition(data, workers, method, seed=0, cutoff=None):
    """Partition the initial training graph; hash is explicitly not METIS."""
    if workers == 1:
        return np.zeros(data.num_nodes, dtype=np.int64)
    if method == "hash":
        return np.random.default_rng(seed).integers(workers, size=data.num_nodes, dtype=np.int64)
    if method != "metis":
        raise ValueError(f"Unknown partition method: {method}")
    import pymetis
    from scipy.sparse import coo_matrix

    # Initial 10% of training interactions: no future/test topology in METIS.
    train_events = np.flatnonzero(data.split == 0)
    initial = (train_events[:max(1, len(train_events) // 10)] if cutoff is None
               else train_events[data.times[train_events] <= cutoff])
    src, dst = data.src[initial], data.dst[initial]
    adjacency = coo_matrix((np.ones(2 * len(src), dtype=np.int32),
                           (np.r_[src, dst], np.r_[dst, src])),
                          shape=(data.num_nodes, data.num_nodes)).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    result = pymetis.part_graph(workers, adjacency=pymetis.CSRAdjacency(
        adjacency.indptr.astype(np.int64), adjacency.indices.astype(np.int64)),
        options=pymetis.Options(seed=seed))
    return np.asarray(result.vertex_part, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=[*JODIE, "synthetic", "products-t"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--events", type=int, default=4096)
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--task", choices=["link", "node"], default="link")
    args = parser.parse_args()
    if (Path(args.output) / "metadata.json").exists():
        raise FileExistsError(f"{args.output} exists; reuse it or choose a new output path")
    if args.name == "synthetic":
        synthetic(args.output, args.seed, args.events, args.nodes, args.task)
    elif args.name == "products-t":
        prepare_products(args.output, args.raw_dir, args.seed)
    else:
        prepare_jodie(args.name, args.output, args.raw_dir)
    print((Path(args.output) / "metadata.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
