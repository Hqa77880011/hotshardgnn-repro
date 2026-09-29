"""Executable mutable-state protocol model for ordered handoff/failure tests.

The distributed training backend uses immutable feature rows and a global barrier;
this separate model exercises delta logging and retained old-epoch reads.
"""

from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class Record:
    value: dict
    owner: int
    epoch: int = 0
    sequence: int = 0
    target: int | None = None
    snapshot: dict | None = None
    deltas: list = field(default_factory=list)
    retired: dict = field(default_factory=dict)


class VersionedStore:
    """Single-threaded state machine; callers provide the routing/write barrier."""

    def __init__(self, workers, log_capacity=1024):
        if workers < 1 or log_capacity < 0:
            raise ValueError("Invalid worker count or delta-log capacity")
        self.workers = workers
        self.log_capacity = log_capacity
        self.records = {}

    def add(self, node, owner, value):
        if node in self.records or not 0 <= owner < self.workers:
            raise ValueError("Duplicate node or invalid owner")
        self.records[node] = Record(deepcopy(value), owner)

    def begin(self, node, target):
        r = self.records[node]
        if r.target is not None or target == r.owner or not 0 <= target < self.workers:
            raise ValueError("Invalid migration target or migration already active")
        r.target, r.snapshot = target, deepcopy(r.value)
        r.deltas = []

    def write(self, node, owner, epoch, sequence, key, value):
        r = self.records[node]
        if (owner, epoch) != (r.owner, r.epoch):
            raise ValueError("Stale epoch or non-authoritative writer")
        if sequence != r.sequence + 1:
            raise ValueError("Updates must be ordered and exactly once")
        r.value[key] = deepcopy(value)
        r.sequence = sequence
        if r.target is not None:
            if len(r.deltas) >= self.log_capacity:
                self.abort(node)  # The committed source update remains valid.
            else:
                r.deltas.append((sequence, key, deepcopy(value)))

    def publish(self, node):
        """Replay the final log at the write barrier, then switch authority."""
        r = self.records[node]
        if r.target is None:
            raise ValueError("No prepared migration")
        complete = deepcopy(r.snapshot)
        for _, key, value in r.deltas:
            complete[key] = deepcopy(value)
        if complete != r.value:
            raise ValueError("Target contents do not match committed source state")
        r.retired[r.epoch] = (deepcopy(r.value), set())
        r.value, r.owner, r.epoch = complete, r.target, r.epoch + 1
        r.target, r.snapshot, r.deltas = None, None, []

    def abort(self, node):
        r = self.records[node]
        r.target, r.snapshot, r.deltas = None, None, []

    def read(self, node, epoch):
        r = self.records[node]
        if epoch == r.epoch:
            return deepcopy(r.value)
        return deepcopy(r.retired[epoch][0])

    def acknowledge(self, node, epoch, worker):
        if not 0 <= worker < self.workers:
            raise ValueError("Invalid worker")
        r = self.records[node]
        _, acknowledgements = r.retired[epoch]
        acknowledgements.add(worker)
        if len(acknowledgements) == self.workers:
            del r.retired[epoch]
