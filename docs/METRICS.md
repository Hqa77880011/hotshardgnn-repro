# 数据、测量与检查

## 输入与采样

JODIE CSV 的 user / item 独立映射至连续的两个节点 ID 区间。保持原 timestamp，统一减去最早时间；同一 timestamp 的事件保持原顺序。按 timestamp 的 70% / 85% 分位点划分 train / validation / test，时间相同的事件不拆到不同划分。

原始 JODIE 特征是 **edge features**。节点特征按 TGAT 官方预处理思路填零，维度为 `max(edge_dim, 32)`。特征是否全零不改变传输计量。合成测试数据使用非零随机特征，用于检查迁移后的内容。

每个链接训练 batch 有等量正例和负例，negative destination 从全部 item 中抽取，排除当前正例。这是 transductive evaluation，不是 TGAT 原论文的 unseen-node inductive protocol。所有可见历史邻居必须满足 `edge_time < query_time`，递归子查询使用该历史事件的时间作为 cutoff。

一个全局 batch 按索引 stride 分配给 consumer rank，consumer 分配不会随所有权变化。尾 batch 不丢弃；没有真实样本的 rank 执行零权重计算以参与 collective，损失按真实全局样本数归一化。训练、验证和测试分别使用独立 seed 派生的 negative RNG。

Products-T 使用 OGB 官方 train / valid / test node indices。合成边按固定 seed 均匀随机排列后赋整数到达时间。每次训练遍历的 cutoff 随 global step 递增，最终评估使用全部历史结构。该时间生成与截止规则是可复查的重建选择。

## 测量

| 字段 | 定义 |
|---|---|
| `throughput` | 测量窗口真实 processed examples / 窗口 wall-clock 秒数之和 |
| `seconds` | 采样、远程读取、forward/backward、优化、遥测、控制和迁移；窗口末端同步 |
| `p95_batch_ms` | 每个 global step 先取所有 rank 的最大耗时，再对所有测量 step 取第 95 百分位 |
| `remote_feature_bytes` | 去重后的非本地 float32 feature rows 响应 payload；无跨 batch cache |
| `remote_demand_proxy` | Eq. (1)，包含全部 state charge，与上一项有不同单位含义 |
| `load_cv_squared` | owner-side fetch 次数 + consumer seed 数的工作模型 CV²，不是 GPU utilization |
| `max_mean_load` | 同一工作模型的 max / mean |
| `scheduled_bytes` | 所选迁移 state charge 总和，用于硬预算检查 |
| `migration_payload_bytes` | 实际发送的 node IDs + feature rows；不包含 barrier、routing 广播或传输协议头 |
| `migration_pause_ms` | 迁移方法执行时间，包含 copy 和发布/ACK barrier |
| `controller_ms` | 预测、规划、计划广播与耗时归约的 elapsed time |
| `test.average_precision` | 收集全部 rank 正负例后统一计算 AP，避免平均 batch AP |
| `test.accuracy` | Products-T 或 node fixture 的全测试集准确率 |

训练 loss 是真实 backward loss 对真实样本数的归一化汇总。窗口日志在计时结束后落盘；数据加载、模型初始化、末尾评估和 checkpoint 保存不计入系统吞吐。

最终评估使用完成规定训练窗口数后的模型，不按测试集指标选择 checkpoint。末尾 checkpoint 保存最终模型与优化器；当前 CLI 不提供中断续跑，续跑还需要恢复采样步和控制器历史。

## 验收次序

1. `pytest` 检查增益公式、预算、load、cooldown、delta-log 和时间截止。
2. `verify_distributed` 必须至少 2 个进程、发生非零强制迁移，并逐项匹配优化器续算。
3. `smoke` 完成全部窗口并输出完整 `summary.json`。
4. `report` 重新核算日志、预算、batch 数和全局 P95，再匹配静态基线。
5. 同 seed 的 placement 对照使用相同计算路径。多次运行后比较系统指标及学习指标，记录硬件和浮点精度设置。

生成请求轨迹可独立运行：

```bash
python -m hotshardgnn.replay --output outputs/replay.csv --seed 7 --budget-bytes 8192
```

此 CSV 的每一行先记录当前窗口的成本，再选择下一窗口的 ownership。它是控制器实验，不产生 throughput / GPU latency 数据。
