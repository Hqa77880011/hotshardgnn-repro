# 基础验证记录

检查日期：2026-09-29。

本地环境为 Windows、Python 3.13.5、PyTorch 2.7.1+cu118、NumPy 2.1.3。以下分布式检查使用两个 CPU/Gloo 进程。

| 检查 | 结果 |
|---|---|
| 控制器、协议与因果采样单元测试 | 10 项通过 |
| 强制迁移 | 128 个节点，9,216 bytes feature / ID payload |
| 迁移后的特征与 owner | 特征逐元素一致；每个节点一个 owner |
| 迁移后的训练续算 | loss、梯度、参数及 Adam 状态一致 |
| 小规模静态与 HotShardGNN 训练 | 两组均完成 3 个预热窗口和 4 个测量窗口 |
| 结果汇总 | 两组日志通过校验，生成 CSV、Markdown 和曲线 |
| Wikipedia 数据准备 | 9,227 节点、157,474 条交互，与论文表中的数量一致 |
| 仓库与打包 | Python 语法、TOML、JSON 和本地文档链接检查通过；wheel 构建成功 |

小规模训练数据上，控制器未选择迁移；迁移功能由上面的强制迁移检查覆盖。该记录不包含性能结论。

Linux GitHub Actions 流程已配置。多 GPU、多机、完整数据集实验及 Products-T 训练尚未执行。Windows 本地未安装 METIS 编译依赖，METIS 路径留待 Linux 环境检查。
