"""Two CPU/Gloo workers, hash ownership, immutable feature migration, causal temporal attention; smoke.json budgets with paired static/hotshard runs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def build_steps(args):
    output = (ROOT / args.output).resolve()
    return [
        ["-m", "hotshardgnn.data", "synthetic", "--seed", "7", "--output", str(output / "data")],
        ["scripts/launch_cpu.py", "--nproc", "2", "hotshardgnn.train", "--config", "configs/smoke.json", "--data", str(output / "data"), "--policy", "static", "--output", str(output / "static")],
        ["scripts/launch_cpu.py", "--nproc", "2", "hotshardgnn.train", "--config", "configs/smoke.json", "--data", str(output / "data"), "--policy", "hotshard", "--output", str(output / "hotshard")],
        ["-m", "hotshardgnn.report", str(output / "static"), str(output / "hotshard"), "--output", str(output / "report")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='runs/fixed-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")

    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
