# Linux 多 GPU / 多机运行

## 单机多 GPU

在仓库根目录激活环境，先完成 README 的数据准备和 CPU 验证：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
torchrun --standalone --nproc_per_node=4 -m hotshardgnn.train \
  --config configs/temporal.json --policy static --output outputs/wiki-static
torchrun --standalone --nproc_per_node=4 -m hotshardgnn.train \
  --config configs/temporal.json --policy hotshard --output outputs/wiki-hotshard
python -m hotshardgnn.report outputs/wiki-static outputs/wiki-hotshard \
  --output outputs/wiki-report --plot
```

每个 rank 对应一张 GPU。CUDA 使用 NCCL 同步梯度，Gloo 在 CPU 上进行特征传输和控制通信。`CUBLAS_WORKSPACE_CONFIG=:4096:8` 由程序在 CUDA 初始化前设置，便于确定性检查。

## 两台服务器，每台两张 GPU

下例使用 `10.0.0.10` 表示首台服务器的内网 IP，运行前替换为实际地址。各节点读取相同的数据、代码和配置，可使用共享文件系统，也可在各自磁盘的相同路径放置数据。输出由 global rank 0 写入。

服务器 0：

```bash
export NNODES=2 NODE_RANK=0 GPUS_PER_NODE=2
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
bash scripts/train_cluster.sh --config configs/temporal.json \
  --policy static --output outputs/wiki-2node-static
```

服务器 1 同时执行：

```bash
export NNODES=2 NODE_RANK=1 GPUS_PER_NODE=2
export MASTER_ADDR=10.0.0.10 MASTER_PORT=29500
bash scripts/train_cluster.sh --config configs/temporal.json \
  --policy static --output outputs/wiki-2node-static
```

静态作业结束后，在两台服务器上将 `--policy` 改为 `hotshard`，输出改为 `outputs/wiki-2node-hotshard`，再启动下一次作业。

主比较是全局 4 workers。扩展到 2、8 workers 时调整 `NNODES × GPUS_PER_NODE`，保持每 worker batch size 并记录总 batch size 的变化。

## 网络与资源

- `MASTER_PORT` 用于 rendezvous；Gloo / NCCL 还需要节点之间建立数据连接。确保所选训练网络节点互通。
- 多网卡机器按训练网络的实际接口设置 `GLOO_SOCKET_IFNAME`、`NCCL_SOCKET_IFNAME`。
- 多机先运行 `hotshardgnn.verify_distributed`：将脚本中的训练模块换成该验证模块，或者直接用相同 `torchrun` 参数启动它。它使用 CPU/Gloo，并强制真实迁移。
- 当前采样是 Python CPU 实现，整个 temporal adjacency index 与 edge features 在每个 rank 内存中各一份。Products-T 尤其需要预估每台机器多 rank 的总内存占用。
- 训练异常后检查最早报错的 rank；没有 `summary.json` 的输出是不完整实验，不应进入结果报告。重新运行时选新的输出目录。

## Windows 本地验证

目标部署环境为 Linux。部分 Windows PyTorch wheel 的 `torchrun` rendezvous 缺少 libuv，即使设置 `USE_LIBUV=0`，launcher 仍可能报错。可用仓库提供的 `python scripts/launch_cpu.py --nproc 2 ...` 启动本机 CPU 验证，它直接建立 `env://` process group；不是多机 launcher。

```powershell
python scripts/launch_cpu.py --nproc 2 hotshardgnn.verify_distributed --data data/synthetic
python scripts/launch_cpu.py --nproc 2 hotshardgnn.train --config configs/smoke.json --policy static --output outputs/win-static
```
