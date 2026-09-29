"""Validate recorded measurements and compare matched runs without paper constants."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_run(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    windows = [json.loads(line) for line in (path / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    moves = [json.loads(line) for line in (path / "moves.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = config["warmup_windows"] + config["measurement_windows"]
    if len(windows) != expected or len(moves) != expected:
        raise ValueError(f"{path}: incomplete window/move log")
    for i, (window, plan) in enumerate(zip(windows, moves)):
        if window["window"] != i or plan["window"] != i:
            raise ValueError(f"{path}: non-sequential window IDs")
        size = sum(m["size"] for m in plan["moves"])
        if (size != window["scheduled_bytes"]
                or (config["policy"] != "periodic" and size > config["budget_bytes"])):
            raise ValueError(f"{path}: inconsistent or over-budget migration")
        if len({m["node"] for m in plan["moves"]}) != len(plan["moves"]):
            raise ValueError(f"{path}: duplicate node in migration plan")
        if i < config["warmup_windows"] and size:
            raise ValueError(f"{path}: migration during static calibration")
    measured = windows[config["warmup_windows"]:]
    for key in ("processed", "seconds", "remote_feature_bytes", "scheduled_bytes", "migration_payload_bytes", "moved_nodes"):
        if not np.isclose(summary[key], sum(w[key] for w in measured), rtol=1e-9, atol=1e-9):
            raise ValueError(f"{path}: summary {key} disagrees with raw windows")
    times = np.load(path / "batch_seconds.npy", allow_pickle=False)
    if len(times) != config["measurement_windows"] * config["window_batches"]:
        raise ValueError(f"{path}: incomplete mini-batch timings")
    if not np.isfinite(times).all() or (times <= 0).any():
        raise ValueError(f"{path}: invalid mini-batch timings")
    if not np.isclose(summary["p95_batch_ms"], np.percentile(times, 95) * 1000):
        raise ValueError(f"{path}: incorrect P95 aggregation")
    if not np.isclose(summary["throughput"], summary["processed"] / summary["seconds"]):
        raise ValueError(f"{path}: incorrect throughput denominator")
    return dict(path=str(path), config=config, summary=summary, windows=windows)


def match_key(run):
    cfg = run["config"]
    # Only placement policy, its weights and migration budget may differ.
    invariant = {key: value for key, value in cfg.items()
                 if key not in {"policy", "controller", "budget_bytes"}}
    return json.dumps(invariant, sort_keys=True)


def comparisons(runs):
    baseline = {}
    for run in runs:
        if run["config"]["policy"] == "static":
            key = match_key(run)
            if key in baseline:
                raise ValueError("Multiple static baselines match the same seed/config; choose one")
            baseline[key] = run
    rows = []
    for run in runs:
        if run["config"]["policy"] == "static":
            continue
        key = match_key(run)
        if key not in baseline:
            raise ValueError(f"No matched static baseline for {run['path']}")
        current, base = run["summary"], baseline[key]["summary"]
        metric = "average_precision" if "average_precision" in current["test"] else "accuracy"
        rows.append(dict(run=run["path"], dataset=current["dataset"], policy=current["policy"],
                         partition=run["config"]["partition"], seed=current["seed"],
                         workers=current["workers"], budget_bytes=run["config"]["budget_bytes"],
                         throughput_speedup=current["throughput"] / base["throughput"],
                         remote_reduction_pct=100 * (1 - current["remote_feature_bytes"] / base["remote_feature_bytes"])
                         if base["remote_feature_bytes"] else None,
                         p95_reduction_pct=100 * (1 - current["p95_batch_ms"] / base["p95_batch_ms"]),
                         quality_metric=metric,
                         quality_delta_pp=100 * (current["test"][metric] - base["test"][metric]),
                         scheduled_bytes=current["scheduled_bytes"], moved_nodes=current["moved_nodes"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    runs = [read_run(path) for path in args.runs]
    rows = comparisons(runs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    fields = ["dataset", "policy", "seed", "workers", "throughput", "p95_batch_ms", "remote_feature_bytes",
              "scheduled_bytes", "migration_payload_bytes", "moved_nodes"]
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "partition", *fields])
        writer.writeheader()
        for run in runs:
            writer.writerow(dict(run=run["path"], partition=run["config"]["partition"],
                                 **{key: run["summary"][key] for key in fields}))
    if rows:
        with (output / "comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Measured reproduction results", "",
             "All values below were recomputed from the supplied run directories. "
             "Synthetic inputs are correctness fixtures, not paper benchmark results.", "",
             "| Dataset / policy / partition / workers / budget | Seeds | Speedup mean ± sample SD | Quality Δ (pp) mean |",
             "|---|---:|---:|---:|"]
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "policy", "partition", "workers", "budget_bytes"))
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        seeds = [r["seed"] for r in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Repeated seeds within report group: {key}")
        values = [r["throughput_speedup"] for r in group]
        sd = f"{np.std(values, ddof=1):.4f}" if len(values) > 1 else "n/a"
        quality = np.mean([r["quality_delta_pp"] for r in group])
        lines.append(f"| {' / '.join(map(str, key))} | {len(seeds)} | {np.mean(values):.4f} ± {sd} | {quality:.6f} |")
    lines += ["", "P95 is computed from all measured global-step latencies, not the mean of window P95s.",
              "Speedup is paired with the static run having the same data, model, partition, seed, world size and window configuration."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for run in runs:
            data = [w for w in run["windows"] if w["phase"] == "measurement"]
            label = f"{run['config']['policy']}, seed {run['config']['seed']}, B={run['config']['budget_bytes']}"
            axes[0].plot([w["window"] for w in data], [w["throughput"] for w in data], label=label)
            axes[1].plot([w["window"] for w in data], [w["remote_feature_bytes"] / 1e6 for w in data])
        axes[0].set(ylabel="Measured examples / second", xlabel="Control window")
        axes[1].set(ylabel="Remote feature payload (MB)", xlabel="Control window")
        axes[0].legend(fontsize=6)
        for axis in axes:
            axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / "timeline.png", dpi=180)
        plt.close(fig)
    print(f"Validated {len(runs)} runs; wrote {output / 'report.md'}")


if __name__ == "__main__":
    main()
