"""Real CPU feature transport over Gloo, with barrier-based owner publication."""

from datetime import timedelta
import os

import numpy as np
import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, device="cpu"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}" if device == "cuda" else "cpu")
        if device == "cuda":
            torch.cuda.set_device(self.device)
        self.group = None
        if self.world > 1:
            dist.init_process_group("nccl" if device == "cuda" else "gloo",
                                    timeout=timedelta(minutes=10))
            self.group = dist.new_group(backend="gloo") if device == "cuda" else dist.group.WORLD

    def barrier(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self.world > 1:
            dist.barrier(group=self.group)

    def sum(self, array):
        tensor = torch.as_tensor(array).clone()
        if self.world > 1:
            dist.all_reduce(tensor, group=self.group)
        return tensor.numpy()

    def maximum(self, value):
        tensor = torch.tensor(float(value), dtype=torch.float64)
        if self.world > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.group)
        return tensor.item()

    def broadcast(self, value):
        objects = [value]
        if self.world > 1:
            dist.broadcast_object_list(objects, src=0, group=self.group)
        return objects[0]

    def gather(self, value):
        objects = [None] * self.world
        if self.world > 1:
            dist.all_gather_object(objects, value, group=self.group)
        else:
            objects[0] = value
        return objects

    def close(self):
        if self.world > 1:
            dist.destroy_process_group()


def exchange(parts, runtime):
    """Exchange variable-length CPU tensors; no pickled feature payloads."""
    if runtime.world == 1:
        return [parts[0].clone()]
    counts = torch.tensor([len(p) for p in parts], dtype=torch.int64)
    all_counts = [torch.empty_like(counts) for _ in parts]
    dist.all_gather(all_counts, counts, group=runtime.group)
    result = [torch.empty((int(all_counts[p][runtime.rank]), *parts[0].shape[1:]),
                          dtype=parts[0].dtype) for p in range(runtime.world)]
    operations = []
    for peer in range(runtime.world):
        if peer == runtime.rank:
            result[peer].copy_(parts[peer])
            continue
        if len(result[peer]):
            operations.append(dist.P2POp(dist.irecv, result[peer], peer, group=runtime.group))
        if len(parts[peer]):
            operations.append(dist.P2POp(dist.isend, parts[peer].contiguous(), peer, group=runtime.group))
    if operations:
        for request in dist.batch_isend_irecv(operations):
            request.wait()
    return result


class ShardedFeatureStore:
    """Each rank owns only its feature rows; fetches and moves cross processes."""

    def __init__(self, features, owners, runtime):
        self.runtime = runtime
        self.owners = np.asarray(owners, dtype=np.int64).copy()
        self.ids = np.flatnonzero(self.owners == runtime.rank)
        self.values = torch.from_numpy(np.asarray(features[self.ids]).copy())
        self.dim = features.shape[1]
        self.epoch = 0
        self.row = np.full(len(owners), -1, dtype=np.int64)
        self.row[self.ids] = np.arange(len(self.ids))
        self.requests = np.zeros(len(owners), dtype=np.int64)
        self.remote_bytes = 0

    @property
    def state_sizes(self):
        # Float32 row plus node identifier and ownership epoch.
        return np.full(len(self.owners), self.dim * 4 + 16, dtype=np.int64)

    def _local(self, ids):
        rows = self.row[ids.numpy()]
        if (rows < 0).any():
            raise RuntimeError("Request routed to a non-owner")
        return self.values[torch.as_tensor(rows)].contiguous()

    def fetch(self, nodes):
        """Deduplicate within a batch, fetch from actual owners, restore order."""
        nodes = np.asarray(nodes, dtype=np.int64)
        unique, inverse = np.unique(nodes, return_inverse=True)
        self.requests[unique] += 1
        requests = [torch.as_tensor(unique[self.owners[unique] == p].copy())
                    for p in range(self.runtime.world)]
        received = exchange(requests, self.runtime)
        answers = exchange([self._local(ids) for ids in received], self.runtime)
        full = torch.empty((len(unique), self.dim), dtype=self.values.dtype)
        for peer, answer in enumerate(answers):
            positions = np.flatnonzero(self.owners[unique] == peer)
            full[torch.as_tensor(positions)] = answer
            if peer != self.runtime.rank:
                self.remote_bytes += answer.numel() * answer.element_size()
        return full[torch.as_tensor(inverse)]

    def telemetry(self):
        demand = np.zeros((len(self.owners), self.runtime.world), dtype=np.int64)
        demand[:, self.runtime.rank] = self.requests
        total = self.runtime.sum(demand)
        self.requests.fill(0)
        return total

    def migrate(self, moves):
        """Transfer complete immutable rows, publish at a barrier, reclaim at ACK.

        Training has stopped at this boundary, so there are no in-flight writes.
        All ranks receive the same plan before calling this collective method.
        """
        seen = set()
        for move in moves:
            if (move.node in seen or self.owners[move.node] != move.source
                    or not 0 <= move.target < self.runtime.world or move.target == move.source):
                raise ValueError("Invalid or stale migration plan")
            seen.add(move.node)
        self.runtime.barrier()
        outgoing = [torch.tensor([m.node for m in moves if m.source == self.runtime.rank and m.target == p],
                                 dtype=torch.int64) for p in range(self.runtime.world)]
        incoming_ids = exchange(outgoing, self.runtime)
        incoming_values = exchange([self._local(ids) for ids in outgoing], self.runtime)
        new_ids = np.concatenate([ids.numpy() for ids in incoming_ids])
        new_values = torch.cat(incoming_values)
        # Stage before publication; source rows still exist until all ranks ACK.
        self.runtime.barrier()
        for move in moves:
            self.owners[move.node] = move.target
        if moves:
            self.epoch += 1
        self.runtime.barrier()
        keep = torch.as_tensor(self.owners[self.ids] == self.runtime.rank)
        self.ids = np.r_[self.ids[keep.numpy()], new_ids]
        self.values = torch.cat([self.values[keep], new_values])
        self.row.fill(-1)
        self.row[self.ids] = np.arange(len(self.ids))
        # Count actual application payload (ID + feature), not TCP/NCCL wire overhead.
        return sum(len(ids) for ids in outgoing) * (8 + self.dim * 4)
