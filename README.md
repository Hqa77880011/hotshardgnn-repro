# HotShardGNN

**Migration-Budgeted Hotspot Repartitioning for Distributed Temporal Graph Neural Network Training**

HotShardGNN 根据训练过程中的访问需求调整节点归属，在每个控制窗口的迁移预算内改善数据局部性和工作负载分布。本仓库实现需求预测、热点筛选、迁移选择和版本化所有权切换，并提供 PyTorch 分布式训练入口。

支持 JODIE 数据集上的时间链接预测，以及 Products-T 上的节点分类。训练使用 `torchrun` 启动，可部署在 Linux 单机多 GPU 或多台服务器上。

## 安装

建议使用 Python 3.11 或 3.12。以下命令均在仓库根目录执行。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

GPU 训练先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)安装与服务器 CUDA 环境匹配的版本，再安装项目：

```bash
python -m pip install -e '.[dev,metis,plots]'
```

只运行 CPU 检查时可使用：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,plots]'
```

`metis` 用于初始图分区和周期性重分区；快速测试使用 Hash 分区，无需安装它。各服务器应使用相同的依赖版本和配置。

## 快速开始

### 1. 检查控制器和迁移协议

```bash
python -m pytest -q
python -m hotshardgnn.data synthetic --output data/synthetic

torchrun --standalone --nproc_per_node=2 \
  -m hotshardgnn.verify_distributed --data data/synthetic
```

验证程序将节点移动到下一进程，检查特征、节点归属，以及迁移前后训练续算的 loss、梯度、参数和 Adam 状态。成功时输出 `"status": "passed"`；`forced_moves` 和 `payload_bytes` 分别记录移动节点数与传输字节数。

### 2. 运行小规模训练对照

```bash
torchrun --standalone --nproc_per_node=2 -m hotshardgnn.train \
  --config configs/smoke.json --policy static \
  --output outputs/smoke-static

torchrun --standalone --nproc_per_node=2 -m hotshardgnn.train \
  --config configs/smoke.json --policy hotshard \
  --output outputs/smoke-hotshard
```

该配置在 CPU 上运行，包含 3 个预热窗口和 4 个测量窗口，每窗口 2 个训练步。合成数据用于检查运行流程，性能评估使用后面的数据集配置。输出目录需使用新路径，程序会拒绝覆盖已有实验。

### 3. 汇总输出

```bash
python -m hotshardgnn.report \
  outputs/smoke-static outputs/smoke-hotshard \
  --output outputs/smoke-report --plot
```

结果目录包含 `report.md`、`runs.csv`、`comparisons.csv` 和 `timeline.png`。汇总程序会核对窗口记录、迁移预算和 batch 计时，再匹配相同配置及种子的静态基线。

Windows 用户可按 [本地启动说明](docs/CLUSTER.md#windows-本地验证)使用 `scripts/launch_cpu.py`。

## 数据准备

### JODIE

```bash
python -m hotshardgnn.data wikipedia --output data/wikipedia
```

将 `wikipedia` 替换为 `reddit`、`mooc` 或 `lastfm` 可准备其余数据集。下载来源为 [JODIE 官方页面](https://snap.stanford.edu/jodie/)，原始文件保存在 `data/raw/`。

预处理将 user 和 item 映射到互不重叠的节点编号，保留边特征，并按时间戳划分 70% / 15% / 15% 的训练、验证和测试集。同一时间戳的事件属于同一划分。采样只访问严格早于查询时间的历史事件。

处理后的目录包含节点特征、边特征、端点、时间戳、划分索引及 `metadata.json`。数据已准备好时，后续训练直接复用该目录。

### Products-T

```bash
python -m pip install -e '.[products]'
python -m hotshardgnn.data products-t --output data/products-t --seed 7
```

Products-T 基于 [OGBN-Products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products)，保留官方节点分类划分。无向边按固定种子排列后赋予合成时间戳，具体规则见 [复现说明](docs/REPRODUCTION.md)。该数据集较大，下载和预处理需要较多磁盘与主机内存。

## 分布式训练

### 单机四卡

先用较短的控制窗口检查环境：

```bash
torchrun --standalone --nproc_per_node=4 -m hotshardgnn.train \
  --config configs/temporal.json --data data/wikipedia \
  --policy static --window-batches 20 --output outputs/wiki-static

torchrun --standalone --nproc_per_node=4 -m hotshardgnn.train \
  --config configs/temporal.json --data data/wikipedia \
  --policy hotshard --window-batches 20 --output outputs/wiki-hotshard
```

`--nproc_per_node` 对应本机使用的 GPU 数。去掉 `--window-batches 20` 后，每个控制窗口按配置运行 2,000 步。训练包含 3 个预热窗口和 12 个测量窗口；数据按时间顺序循环，累计遍数记录在 `summary.json` 的 `training_passes` 中。

CPU 上运行真实数据的短流程可使用 `configs/quick.json`。Products-T 将配置和数据路径替换为 `configs/products.json` 与 `data/products-t`，训练入口会选择 GraphSAGE。

### 多台服务器

每台服务器设置自己的 `NODE_RANK`，使用相同的 `NNODES`、`MASTER_ADDR` 和 `MASTER_PORT`：

```bash
export NNODES=2 NODE_RANK=0 GPUS_PER_NODE=2
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500

bash scripts/train_cluster.sh --config configs/temporal.json \
  --policy hotshard --output outputs/wiki-2node-hotshard
```

将示例地址换成首台服务器可达的内网地址，第二台服务器设置 `NODE_RANK=1`。数据与配置路径在各节点保持一致，输出由 global rank 0 写入。网络接口与完整启动示例见 [集群指南](docs/CLUSTER.md)。

## 配置与对照方法

| 配置 | 用途 | 默认设备 | 初始分区 |
|---|---|---|---|
| `configs/smoke.json` | 合成数据运行检查 | CPU | Hash |
| `configs/quick.json` | 真实数据短流程 | CPU | METIS |
| `configs/temporal.json` | 时间链接预测 | CUDA | METIS |
| `configs/products.json` | Products-T 节点分类 | CUDA | METIS |

JSON 中可以设置 batch size、fanout、学习率、窗口长度、迁移预算和控制器参数。`--data`、`--device`、`--partition`、`--policy`、`--seed`、`--budget-bytes` 和 `--window-batches` 会覆盖对应配置；最终参数保存到运行目录的 `config.json`。

| `--policy` | 行为 |
|---|---|
| `static` | 保留初始分区 |
| `hotshard` | 完整控制器 |
| `periodic` | 每第三个测量窗口重新运行 METIS 并迁移变更节点 |
| `no_forecast` | 使用当前窗口需求，ρ=1 |
| `cut_only` | 仅保留 locality benefit，β=0 |
| `load_only` | 仅保留 load benefit，α=0 |
| `no_penalty` | 关闭迁移代价项，γ=0 |
| `no_cooldown` | 关闭迁移冷却期，K=0 |
| `random` | 对相同可行候选随机排序 |

`periodic` 使用 METIS 初始分区，不受局部迁移预算限制。`static` 的基线名称取决于 `--partition`：使用 Hash 时应报告 Static Hash，使用 METIS 时报告 Static METIS。

批量实验脚本包含五个种子的静态对照、组件消融和预算扫描：

```bash
bash scripts/run_suite.sh data/wikipedia outputs/wiki-suite 4
python -m hotshardgnn.report outputs/wiki-suite/* \
  --output outputs/wiki-report --plot
```

种子固定为 `7, 17, 27, 37, 47`。可以通过 `CONFIG` 和 `WINDOW_BATCHES` 环境变量调整配置，例如：

```bash
WINDOW_BATCHES=20 bash scripts/run_suite.sh data/wikipedia outputs/wiki-short 4
```

## 结果汇总

| 文件 | 内容 |
|---|---|
| `config.json` | 本次运行的完整参数 |
| `environment.json` | 软件版本、设备与数据来源 |
| `windows.jsonl` | 每窗口的吞吐、loss、远程读取、负载和迁移量 |
| `moves.jsonl` | 每窗口发布的节点迁移与评分 |
| `batch_seconds.npy` | 各测量步的最大 rank 延迟 |
| `summary.json` | 汇总指标、验证集与测试集结果 |
| `checkpoint.pt` | 最终模型、优化器与节点归属 |

吞吐按处理样本数除以测量窗口总耗时计算，包含控制器和迁移暂停。P95 基于全部测量步计算。链接预测报告 AP 和 ROC-AUC，节点分类报告 accuracy。

`scheduled_bytes` 是控制器用于预算判断的状态计费；`migration_payload_bytes` 是迁移发送的节点 ID 和特征字节。指标定义与数据处理细节见 [数据与指标](docs/METRICS.md)。

控制器也可以独立运行：

```bash
python -m hotshardgnn.replay --output outputs/replay.csv \
  --seed 7 --budget-bytes 8192
```

该命令生成周期性漂移的请求轨迹，输出 locality cost、load dispersion 和迁移量。

## 代码结构

```text
configs/                     训练与控制器配置
docs/                        实现映射、集群运行和指标说明
scripts/                     批量实验与启动脚本
src/hotshardgnn/
  controller.py              需求预测、候选筛选和预算约束选择
  distributed.py             跨进程特征读取与所有权迁移
  protocol.py                可变状态迁移协议的测试模型
  data.py                    数据准备与图分区
  sampling.py                因果时间采样
  models.py                  时间注意力与 GraphSAGE
  train.py                   分布式训练和窗口计量
  report.py                  日志校验与结果汇总
  replay.py                  控制器请求轨迹实验
  verify_distributed.py      强制迁移与训练一致性检查
tests/                       控制器、协议和采样测试
.github/workflows/ci.yml      Linux CPU 检查流程
```


## 许可与引用

代码采用 [MIT License](LICENSE)。数据集遵循各自的许可要求。

引用本实现可使用 [CITATION.cff](CITATION.cff)。实验中使用 JODIE、TGAT、GraphSAGE 或 OGB 时，请同时引用对应工作，链接见 [复现说明](docs/REPRODUCTION.md)。
