# HotShardGNN

Implementation of **Migration-Budgeted Hotspot Repartitioning for Distributed Temporal Graph Neural Network Training**.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Download and preprocess [JODIE Wikipedia](https://snap.stanford.edu/jodie/) interactions:

```bash
python -m hotshardgnn.data wikipedia --output data/wikipedia --seed 7
```

The same command accepts `reddit`, `mooc` and `lastfm`. Raw files go to `data/raw`; processed arrays and `metadata.json` go to `--output`. Timestamp groups stay together in the 70/15/15 split. For a generated CPU workload:

```bash
python -m hotshardgnn.data synthetic --output data/synthetic --seed 7
```

Products-T uses [OGBN-Products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) with generated timestamps and official node splits:

```bash
python -m pip install -e ".[products]"
python -m hotshardgnn.data products-t --output data/products-t --seed 7
```

## Distributed training

The CPU example uses two Gloo workers and hash ownership. Policies share data, seed and configuration; choose fresh output directories.

```bash
python scripts/launch_cpu.py --nproc 2 hotshardgnn.train --config configs/smoke.json --data data/synthetic --policy static --seed 7 --output outputs/smoke-static
python scripts/launch_cpu.py --nproc 2 hotshardgnn.train --config configs/smoke.json --data data/synthetic --policy hotshard --seed 7 --output outputs/smoke-hotshard
```

For Linux multi-GPU training, install a CUDA-compatible PyTorch build and METIS extra:

```bash
python -m pip install -e ".[metis]"
torchrun --standalone --nproc_per_node=4 -m hotshardgnn.train --config configs/temporal.json --data data/wikipedia --policy static --seed 7 --output outputs/wiki-static
torchrun --standalone --nproc_per_node=4 -m hotshardgnn.train --config configs/temporal.json --data data/wikipedia --policy hotshard --seed 7 --output outputs/wiki-hotshard
```

Products-T uses `configs/products.json` and `data/products-t`. Multi-node setup is in [docs/CLUSTER.md](docs/CLUSTER.md). `--data`, `--device`, `--partition`, `--policy`, `--seed`, `--budget-bytes` and `--window-batches` override JSON settings.

## Evaluation

Training saves validation/test quality and window measurements. Summarize paired runs with:

```bash
python -m pip install -e ".[plots]"
python -m hotshardgnn.report outputs/smoke-static outputs/smoke-hotshard --output outputs/smoke-report --plot
```

## Ablations

`--policy` supports `static`, `hotshard`, `periodic`, `no_forecast`, `cut_only`, `load_only`, `no_penalty`, `no_cooldown` and `random`. The Linux suite runs paired seeds 7/17/27/37/47, component controls and budget scans:

```bash
bash scripts/run_suite.sh data/wikipedia outputs/wiki-suite 4
```

`CONFIG` and `WINDOW_BATCHES` select existing settings. The periodic control uses global METIS repartitioning.

## Outputs

Runs store `config.json`, `environment.json`, `windows.jsonl`, `moves.jsonl`, `batch_seconds.npy`, `summary.json` and `checkpoint.pt`: settings, environment, window measurements, migrations, delays, task metrics and final model/optimizer/ownership. Reports contain `report.md`, `runs.csv`, `comparisons.csv` and optional `timeline.png`.

Inspect the existing CPU pipeline with `python reproduce.py --plan`, or execute it with `python reproduce.py --output runs/fixed-setting`. Historical checks are recorded in [docs/VALIDATION.md](docs/VALIDATION.md); this pass performed only the source/configuration checks above. Code uses the [MIT License](LICENSE); citation metadata is in [CITATION.cff](CITATION.cff).
