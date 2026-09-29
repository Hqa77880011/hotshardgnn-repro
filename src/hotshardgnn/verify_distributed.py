"""Force actual migrations and check features, model/optimizer continuation."""

import argparse
from copy import deepcopy
import json

import numpy as np
import torch

from .controller import Move
from .data import Dataset
from .distributed import Runtime, ShardedFeatureStore
from .models import GraphModel
from .sampling import TemporalSampler, leaf_nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic")
    args = parser.parse_args()
    rt = Runtime("cpu")
    try:
        if rt.world < 2:
            raise ValueError("Launch with torchrun --nproc_per_node=2 (or more)")
        torch.set_num_threads(1)
        torch.manual_seed(11)
        data = Dataset(args.data)
        owners = np.arange(data.num_nodes, dtype=np.int64) % rt.world
        store = ShardedFeatureStore(data.features, owners, rt)
        sampler = TemporalSampler(data)
        ids = np.arange(6) + rt.rank * 6
        tree = sampler.sample(np.r_[data.src[ids], data.dst[ids], data.dst[ids + 6]],
                              np.tile(data.times[ids], 3), [2, 2])
        before = store.fetch(leaf_nodes(tree))
        torch.testing.assert_close(before, torch.from_numpy(data.features[leaf_nodes(tree)].copy()), rtol=0, atol=0)
        model = GraphModel(data.features.shape[1], data.edge_features.shape[1], 8, 2)
        opt = torch.optim.Adam(model.parameters(), lr=.01)

        def step(m, o, features):
            o.zero_grad()
            loss = m(tree, features, data.edge_features, 2).square().mean()
            loss.backward()
            gradients = [p.grad.clone() for p in m.parameters()]
            o.step()
            return loss.detach(), gradients

        step(model, opt, before)
        reference = deepcopy(model)
        reference_opt = torch.optim.Adam(reference.parameters(), lr=.01)
        reference_opt.load_state_dict(deepcopy(opt.state_dict()))
        moves = [Move(node, int(owner), int((owner + 1) % rt.world), int(store.state_sizes[node]), 1.0)
                 for node, owner in enumerate(owners)]
        payload = store.migrate(moves)
        after = store.fetch(leaf_nodes(tree))
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        np.testing.assert_array_equal(store.owners, (owners + 1) % rt.world)
        counts = rt.sum((store.row >= 0).astype(np.int64))
        np.testing.assert_array_equal(counts, np.ones(data.num_nodes, dtype=np.int64))
        before_loss, before_grad = step(reference, reference_opt, before)
        after_loss, after_grad = step(model, opt, after)
        torch.testing.assert_close(before_loss, after_loss, rtol=0, atol=0)
        for a, b in zip(before_grad, after_grad):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(reference.parameters(), model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for key, state in opt.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, reference_opt.state_dict()["state"][key][name], rtol=0, atol=0)
        total_payload = int(rt.sum(np.array(payload, dtype=np.int64)))
        rt.barrier()
        if rt.rank == 0:
            print(json.dumps(dict(status="passed", workers=rt.world, forced_moves=len(moves),
                                  payload_bytes=total_payload, checks=["feature equality", "one owner",
                                  "causal samples", "loss", "gradients", "parameters", "Adam state"]), indent=2))
    finally:
        rt.close()


if __name__ == "__main__":
    main()
