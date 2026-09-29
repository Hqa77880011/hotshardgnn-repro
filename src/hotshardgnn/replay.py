"""CPU-only drifting-demand experiment reporting objective costs and migration."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from .controller import Controller, ControllerConfig, load_cost, remote_cost, worker_loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/replay.csv")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget-bytes", type=int, default=8192)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    nodes, workers = 1024, 4
    initial = np.arange(nodes) % workers
    initial[:80] = 0
    sizes = np.full(nodes, 256, dtype=np.int64)
    policies = {"static": None, "hotshard": ControllerConfig(),
                "no_forecast": replace(ControllerConfig(), rho=1),
                "cut_only": replace(ControllerConfig(), beta=0),
                "load_only": replace(ControllerConfig(), alpha=0),
                "no_penalty": replace(ControllerConfig(), gamma=0),
                "no_cooldown": replace(ControllerConfig(), cooldown=0)}
    owners = {key: initial.copy() for key in policies}
    controllers = {key: Controller(nodes, workers, cfg) for key, cfg in policies.items() if cfg is not None}
    rows = []
    for window in range(15):
        requests = rng.poisson(0.5, (nodes, workers)).astype(float)
        consumer = (window // 3) % workers
        requests[:80, consumer] += rng.poisson(60, 80)
        base = np.full(workers, 1000.0)
        for policy in policies:
            owner = owners[policy]
            row = dict(window=window, phase="warmup" if window < 3 else "measurement", policy=policy,
                       remote_demand_proxy=remote_cost(requests, owner, sizes),
                       load_cv_squared=load_cost(worker_loads(owner, requests.sum(1), base)),
                       scheduled_bytes=0, moved_nodes=0)
            if policy != "static":
                c = controllers[policy]
                c.observe(requests, window)
                if window < 3:
                    c.calibrate(requests, owner, sizes, base)
                    if window == 2:
                        c.freeze_scales()
                else:
                    moves = c.plan(requests, owner, sizes, base, args.budget_bytes, window)
                    for move in moves:
                        owner[move.node] = move.target
                    c.commit(moves, window)
                    row.update(scheduled_bytes=sum(m.size for m in moves), moved_nodes=len(moves))
            rows.append(row)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}. Generated demand trace; no training-throughput claims.")


if __name__ == "__main__":
    main()
