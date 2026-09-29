# 论文到代码的对应关系

本页列出 HotShardGNN 的方法实现、参数来源，以及训练后端与论文设置的差异。

## 已实现的内容

| 论文机制 | 位置 | 实现与验证 |
|---|---|---|
| Eq. (1) remote-demand proxy | `controller.remote_cost` | 按节点状态计费加权；与完整重算的增益对照 |
| Eq. (2) squared CV | `controller.load_cost` | 转移 owner 服务工作，保持 consumer 工作不变 |
| EMA，ρ=0.7 | `Controller.observe` | 每窗口更新，冷记录超过 2K 个窗口后清零 |
| 热度与外部请求筛选 | `Controller.plan` | 当前请求比率、严格高于第 90 百分位、owner 负载高于均值 |
| 目标 consumer 至少 10% | `Controller.plan` | 按当前窗口请求份额筛选，EMA 用于增益预测 |
| 归一化局部 utility | `Controller.plan` | locality、CV 改善、迁移代价及近期惩罚 |
| U / state bytes 贪心 | `Controller.plan` | 每次接纳重算 utility、目标负载及剩余预算 |
| reserve ζ=0.15 | `Controller.plan` | 排名处理进入最后四分之一时释放 |
| 一换二 refinement | `Controller.plan` | 检查最多 64 个拒绝候选的不同节点组合，仅交换一次 |
| K=3 cooldown | `Controller.commit` | 发布成功才更新；随后三个窗口禁止再次移动 |
| 固定 R0 / L0 | `Controller.calibrate` | 三个静态预热窗口取中位数，之后不更新 |
| copy → publish → reclaim | `ShardedFeatureStore.migrate` | 实际进程间传输；两次 barrier 分别界定发布与 ACK |
| 可变状态 delta / epoch | `protocol.VersionedStore` | 独立可执行状态机测试，含溢出、重复写、旧 epoch、延迟 ACK |
| TGAT / GraphSAGE | `models.py` / `sampling.py` | 时间注意力／mean aggregation，严格因果的递归采样 |
| 静态 / periodic / 消融 | `train.py` | 相同初始分区、事件、模型及随机数流 |
| 结果计量与对照 | `report.py` | 从窗口和 batch 记录重新核算，拒绝缺失或不匹配基线 |

训练后端通过 Gloo 读取远端 owner 的特征并发送迁移 payload，吞吐使用 wall-clock 计时。`replay.py` 单独评估生成请求轨迹下的 objective costs 和迁移量。

## 论文明确给出的参数

| 参数 | 默认 |
|---|---:|
| control window | 2,000 个全局 mini-batch 步 |
| warmup / measured windows | 3 / 12 |
| EMA ρ | 0.7 |
| heat percentile | 90 |
| external threshold | 0.30 |
| destination share | 0.10 |
| cooldown K | 3 |
| load slack ε | 0.10 |
| reserve ζ | 0.15 |
| minimum utility μ | 0.01 |
| fixed budgets | 是；未启用可选 feedback |

论文主比较为 4 个 worker；每 worker 配置 1×A100 40GB、32 CPU cores、256GB RAM 和 100Gb/s 网络。

## 重建参数

下列细节未在论文中完整给出，本仓库采用以下配置：

| 缺项 | 本仓库选择 |
|---|---|
| α、β、γ、η 的数值 | 1、1、0.05、0；全部可在 JSON 中调整 |
| 服务工作模型 | 每次去重特征请求一单位 owner work，每条种子一个 consumer work |
| heat 信号组合 | 当前后端使用请求数、远程 feature bytes、feature reads；缺少独立 update/degree-change counters |
| median 为零 | 对该信号使用分母 1；无需求节点不参与 heat percentile |
| R0 / L0 为零 | 下界分别为 1 byte-request / 1e-6 |
| EMA 初值 | 零，依次观察三个预热窗口 |
| 一换二淘汰哪个已接纳 move | 淘汰最低 utility 的一个，再检查最多 64 个拒绝 move 的组合 |
| node state charge | float32 feature row + 16 bytes ID / epoch metadata |
| fixed seed 列表 | 7、17、27、37、47；不是论文原 seed 列表 |
| 数据划分 | JODIE 时间戳 70/15/15 transductive split；Products 保留官方节点划分 |
| negative samples | 均匀抽 destination，排除当前 positive；允许历史正边作为负样本 |
| 时间采样 | 最近 k 个严格早于 cutoff 的事件；每层子查询截止时间为对应历史事件时间 |
| batch size / fanout / hidden / optimizer | 在 JSON 显式列出；TGAT 风格网络为独立重写 |
| 初始 METIS 图 | 训练事件最早 10% 的无向去重图，全部节点保留，包括 isolated nodes |
| Periodic Full | 每第三个测量窗口按已可见训练图重新运行 METIS；迁移全部变化的 owner，免除预算约束 |
| Products-T 时间戳 | canonical 无向边以固定 seed 打乱，排序后赋连续整数时间 |

PyMETIS 的图是无向的；多次 temporal interaction 在分区图中合并，但采样索引保留每次 interaction。OGB 已保存双向边，本仓库先取 `source < target`，采样索引再展开双向，避免把官方无向边数再翻倍。

## 与论文系统实现的差异

1. **后端**：论文描述修改 DGL 分布式存储并接入 TGL；当前实现是 PyTorch DDP + Gloo CPU feature transport，CUDA 训练梯度用 NCCL。
2. **状态范围**：训练迁移只涉及 immutable features。图结构、edge features 和 sampler index 每个 rank 保留一份；TGAT/GraphSAGE 不使用可变 temporal memory。`protocol.py` 对可变状态进行独立逻辑验证，没有宣称实现生产级 mutable-memory RPC。
3. **执行模式**：控制器在窗口边界同步运行，未与末尾 mini-batch 异步重叠。网络使用按需批量 P2P，不实现 DGL 的共享内存计数器、RCU 路由和独立 control stream。
4. **控制器空间**：参考实现保存 dense N×P demand/forecast；冷记录清零而不是 sparse 结构释放。不沿用论文 sparse-memory 复杂度声明。
5. **缓存**：仅 batch 内去重，无跨 batch 缓存。Cache-only、DistDy、MemShare 不在已实现对照中。
6. **故障**：状态机验证发布前丢失/溢出与延迟回收；分布式后端不支持 worker 丢失后继续执行。训练发生错误时作业失败，不输出完整成功汇总。
7. **工作负载**：`replay.py` 的 1,024 节点漂移请求不是论文百万节点、1,600 万边、Zipf 1.2 的动态图生成器；该生成器及原始时序未提供。

性能结果需结合后端、模型设置和硬件条件比较；实验输出由运行日志生成。

## 阅读顺序

1. `controller.py`：公式和 Algorithm 1。
2. `protocol.py` 与对应测试：可变状态交接。
3. `distributed.py`：真实特征请求与迁移。
4. `sampling.py` / `models.py`：固定计算路径。
5. `train.py`：窗口循环、校准、计量、最终评估。
6. `report.py`：基于运行产物的复核与对照。

参考资料：[JODIE](https://snap.stanford.edu/jodie/)、[TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs)、[GraphSAGE](https://snap.stanford.edu/graphsage/)、[PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed.html)、[PyMETIS](https://documen.tician.de/pymetis/functionality.html)、[OGB](https://ogb.stanford.edu/docs/nodeprop/)。
