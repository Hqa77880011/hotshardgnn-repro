"""torchrun entry point: real sharded reads, DDP training and windowed migration."""

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .controller import Controller, ControllerConfig, Move, load_cost, remote_cost, worker_loads
from .data import Dataset, partition
from .distributed import Runtime, ShardedFeatureStore
from .models import GraphModel
from .sampling import TemporalSampler, leaf_nodes


POLICIES = ("static", "periodic", "hotshard", "no_forecast", "cut_only", "load_only",
            "no_penalty", "no_cooldown", "random")


def policy_config(config, policy):
    changes = {"no_forecast": {"rho": 1.0}, "cut_only": {"beta": 0.0},
               "load_only": {"alpha": 0.0}, "no_penalty": {"gamma": 0.0},
               "no_cooldown": {"cooldown": 0}, "random": {"one_swap": False}}
    return replace(config, **changes.get(policy, {}))


def make_batch(data, sampler, indices, runtime, config, step, phase=0):
    """Fixed consumer assignment and per-step negative RNG, independent of owners."""
    local = indices[runtime.rank::runtime.world]
    valid_count = len(local)
    if valid_count == 0:
        local = indices[:1]  # Participate in collectives; loss contribution is zero.
    if data.meta["task"] == "link":
        source, target, times = data.src[local], data.dst[local], data.times[local]
        rng = np.random.default_rng(np.random.SeedSequence([config["seed"], phase, step, runtime.rank]))
        nitems = data.meta["num_items"]
        if nitems < 2:
            raise ValueError("Link prediction requires at least two destination nodes")
        # Uniform negative destination, excluding the current positive only.
        negative = rng.integers(nitems - 1, size=len(local)) + data.meta["num_users"]
        negative += negative >= target
        nodes = np.r_[source, target, negative]
        cutoff = np.tile(times, 3)
        labels = torch.cat([torch.ones(len(local)), torch.zeros(len(local))])
    else:
        nodes = local
        # Synthetic timestamp windows expose successively more edges; repeat per pass.
        train_count = max(1, int((data.node_split == 0).sum()))
        steps_per_pass = math.ceil(train_count / (config["batch_size"] * runtime.world))
        fraction = (step % steps_per_pass + 1) / steps_per_pass if phase == 0 else 1.0
        cutoff = np.full(len(local), fraction * (float(data.times[-1]) + 1))
        labels = torch.from_numpy(np.asarray(data.labels[local]).copy()).long()
    tree = sampler.sample(nodes, cutoff, config["fanouts"])
    return tree, labels.to(runtime.device), valid_count


def evaluate(model, data, sampler, store, runtime, config, split):
    model.eval()
    ids = np.flatnonzero((data.split if data.meta["task"] == "link" else data.node_split) == split)
    if len(ids) == 0:
        raise ValueError(f"Evaluation split {split} is empty")
    labels_all, scores_all = [], []
    batch = config["batch_size"] * runtime.world
    with torch.no_grad():
        for step, start in enumerate(range(0, len(ids), batch)):
            tree, labels, valid = make_batch(data, sampler, ids[start:start + batch], runtime, config, step, split)
            features = store.fetch(leaf_nodes(tree)).to(runtime.device)
            logits = model(tree, features, data.edge_features, len(config["fanouts"]))
            if valid:
                labels_all.extend(labels.cpu().tolist())
                scores_all.extend((logits.sigmoid() if data.meta["task"] == "link"
                                   else logits.argmax(-1)).cpu().tolist())
    gathered = runtime.gather((labels_all, scores_all))
    y = np.concatenate([part[0] for part in gathered])
    score = np.concatenate([part[1] for part in gathered])
    if data.meta["task"] == "link":
        return {"average_precision": float(average_precision_score(y, score)),
                "roc_auc": float(roc_auc_score(y, score)), "examples": len(y)}
    return {"accuracy": float(np.mean(y == score)), "examples": len(y)}


def run(config, output):
    runtime = Runtime(config["device"])
    try:
        return _run(config, Path(output), runtime)
    finally:
        runtime.close()


def _run(config, output, runtime):
    if runtime.rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    runtime.barrier()
    torch.set_num_threads(config.get("cpu_threads", 1))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if runtime.device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    torch.use_deterministic_algorithms(True)
    data = Dataset(config["data"])
    owners = partition(data, runtime.world, config["partition"], config["partition_seed"]) if runtime.rank == 0 else None
    owners = runtime.broadcast(owners)
    sampler = TemporalSampler(data)
    store = ShardedFeatureStore(data.features, owners, runtime)
    classes = int(data.labels.max()) + 1 if data.meta["task"] == "node" else 2
    kind = "tgat" if data.meta["task"] == "link" else "sage"
    model = GraphModel(data.features.shape[1], data.edge_features.shape[1], config["hidden"],
                       len(config["fanouts"]), config["heads"], kind, classes).to(runtime.device)
    wrapped = (DistributedDataParallel(model, device_ids=[runtime.local_rank]
               if runtime.device.type == "cuda" else None) if runtime.world > 1 else model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    cfg = policy_config(ControllerConfig(**config.get("controller", {})), config["policy"])
    controller = Controller(data.num_nodes, runtime.world, cfg) if runtime.rank == 0 else None
    train_ids = np.flatnonzero((data.split if data.meta["task"] == "link" else data.node_split) == 0)
    global_batch = config["batch_size"] * runtime.world
    if len(train_ids) < runtime.world:
        raise ValueError("Training split has fewer examples than workers")
    batches_per_pass = math.ceil(len(train_ids) / global_batch)
    steps = config["window_batches"]
    warmup, measured = config["warmup_windows"], config["measurement_windows"]
    config = dict(config, world_size=runtime.world, model=kind, controller=asdict(cfg))
    if runtime.rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (output / "environment.json").write_text(json.dumps(dict(
            python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            platform=platform.platform(), device=str(runtime.device),
            device_name=torch.cuda.get_device_name(runtime.device) if runtime.device.type == "cuda" else platform.processor(),
            data=data.meta), indent=2) + "\n", encoding="utf-8")
    records, measured_latencies = [], []
    step = 0
    visible_cutoff = -1.0
    for window in range(warmup + measured):
        model.train()
        runtime.barrier()
        window_start = time.perf_counter()
        batch_times, loss_sum, processed = [], 0.0, 0
        local_consumers = np.zeros(runtime.world, dtype=np.float64)
        remote_before = store.remote_bytes
        for _ in range(steps):
            start = (step % batches_per_pass) * global_batch
            ids = train_ids[start:start + global_batch]
            if kind == "tgat":
                visible_cutoff = max(visible_cutoff, float(data.times[ids].max()))
            else:
                fraction = ((step % batches_per_pass) + 1) / batches_per_pass
                visible_cutoff = max(visible_cutoff, fraction * (float(data.times[-1]) + 1))
            began = time.perf_counter()
            tree, labels, valid = make_batch(data, sampler, ids, runtime, config, step)
            features = store.fetch(leaf_nodes(tree)).to(runtime.device)
            optimizer.zero_grad(set_to_none=True)
            logits = wrapped(tree, features, data.edge_features, len(config["fanouts"]))
            raw_loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
                        if kind == "tgat" else F.cross_entropy(logits, labels, reduction="sum"))
            examples = len(ids) * (2 if kind == "tgat" else 1)
            loss = raw_loss * (runtime.world / examples) * bool(valid)
            loss.backward()
            optimizer.step()
            if runtime.device.type == "cuda":
                torch.cuda.synchronize(runtime.device)
            batch_times.append(time.perf_counter() - began)
            loss_sum += float(raw_loss.detach()) * bool(valid)
            processed += len(ids)
            local_consumers[runtime.rank] += valid
            step += 1
        # Telemetry and controller work are inside the measured window.
        requests = store.telemetry()
        consumers = runtime.sum(local_consumers)
        remote = float(runtime.sum(np.array(store.remote_bytes - remote_before, dtype=np.int64)))
        actual_loads = worker_loads(store.owners, requests.sum(1), consumers)
        cost = remote_cost(requests, store.owners, store.state_sizes)
        controller_start = time.perf_counter()
        moves = []
        if runtime.rank == 0:
            controller.observe(requests, window)
            if window < warmup:
                controller.calibrate(requests, store.owners, store.state_sizes, consumers)
                if window == warmup - 1:
                    controller.freeze_scales()
            elif config["policy"] == "periodic":
                if (window - warmup + 1) % 3 == 0:
                    new_owners = partition(data, runtime.world, "metis", config["partition_seed"], visible_cutoff)
                    moves = [Move(int(node), int(store.owners[node]), int(new_owners[node]),
                                  int(store.state_sizes[node]), 0.0)
                             for node in np.flatnonzero(new_owners != store.owners)]
            elif config["policy"] != "static":
                total = requests.sum(1)
                external = total - requests[np.arange(data.num_nodes), store.owners]
                signals = np.column_stack([total, external * store.dim * 4, total])
                moves = controller.plan(requests, store.owners, store.state_sizes, consumers,
                                        config["budget_bytes"], window, signals,
                                        config["seed"] + window if config["policy"] == "random" else None)
        moves = runtime.broadcast(moves)
        controller_seconds = runtime.maximum(time.perf_counter() - controller_start)
        migration_start = time.perf_counter()
        migration_payload = store.migrate(moves) if moves else 0
        migration_seconds = runtime.maximum(time.perf_counter() - migration_start) if moves else 0.0
        if runtime.rank == 0:
            controller.commit(moves, window)
        runtime.barrier()
        elapsed = runtime.maximum(time.perf_counter() - window_start)
        payload = int(runtime.sum(np.array(migration_payload, dtype=np.int64)))
        batch_times = np.max(np.asarray(runtime.gather(batch_times)), axis=0)
        total_loss = float(runtime.sum(np.array(loss_sum)))
        record = dict(window=window, phase="warmup" if window < warmup else "measurement",
                      processed=processed, seconds=elapsed, throughput=processed / elapsed,
                      p95_batch_ms=float(np.percentile(batch_times, 95) * 1000),
                      loss=total_loss / (processed * (2 if kind == "tgat" else 1)),
                      remote_feature_bytes=int(remote), remote_demand_proxy=cost,
                      load_cv_squared=load_cost(actual_loads),
                      max_mean_load=float(actual_loads.max() / actual_loads.mean()),
                      scheduled_bytes=sum(m.size for m in moves), migration_payload_bytes=payload,
                      migration_pause_ms=migration_seconds * 1000,
                      controller_ms=controller_seconds * 1000, moved_nodes=len(moves),
                      ownership_epoch=store.epoch)
        records.append(record)
        if window >= warmup:
            measured_latencies.extend(batch_times.tolist())
        if runtime.rank == 0:
            with (output / "windows.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            with (output / "moves.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"window": window, "moves": [asdict(m) for m in moves]}) + "\n")
            print(f"window={window:02d} {record['phase']} loss={record['loss']:.4f} "
                  f"examples/s={record['throughput']:.1f} moves={len(moves)}", flush=True)
    validation = evaluate(model, data, sampler, store, runtime, config, 1)
    test = evaluate(model, data, sampler, store, runtime, config, 2)
    if runtime.rank == 0:
        measured_records = records[warmup:]
        summary = dict(backend="pytorch-gloo-feature-store", policy=config["policy"],
                       dataset=data.meta["name"], seed=config["seed"], workers=runtime.world,
                       processed=sum(r["processed"] for r in measured_records),
                       seconds=sum(r["seconds"] for r in measured_records),
                       remote_feature_bytes=sum(r["remote_feature_bytes"] for r in measured_records),
                       scheduled_bytes=sum(r["scheduled_bytes"] for r in measured_records),
                       migration_payload_bytes=sum(r["migration_payload_bytes"] for r in measured_records),
                       moved_nodes=sum(r["moved_nodes"] for r in measured_records),
                       p95_batch_ms=float(np.percentile(measured_latencies, 95) * 1000),
                       validation=validation, test=test,
                       training_steps=step, training_passes=step / batches_per_pass,
                       calibration={"r0": controller.r0, "l0": controller.l0},
                       measurement="real wall clock; application payload excludes transport headers")
        summary["throughput"] = summary["processed"] / summary["seconds"]
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        np.save(output / "batch_seconds.npy", np.asarray(measured_latencies))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "owners": torch.from_numpy(store.owners), "config": config}, output / "checkpoint.pt")
        print(json.dumps(summary, indent=2), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--partition", choices=["hash", "metis"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget-bytes", type=int)
    parser.add_argument("--window-batches", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for key in ("data", "policy", "device", "partition", "seed", "budget_bytes", "window_batches"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("batch_size", "window_batches", "warmup_windows", "measurement_windows", "hidden", "heads"):
        if config[key] <= 0:
            parser.error(f"{key} must be positive")
    if not config["fanouts"] or min(config["fanouts"]) <= 0 or config["budget_bytes"] < 0:
        parser.error("fanouts must be positive; budget must be nonnegative")
    if config["hidden"] % config["heads"]:
        parser.error("hidden must be divisible by heads")
    if config["policy"] == "periodic" and config["partition"] != "metis":
        parser.error("Periodic full repartition requires METIS initial placement")
    run(config, args.output)


if __name__ == "__main__":
    main()
