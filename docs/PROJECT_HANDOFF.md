# 项目交接

- 更新时间：2026-10-17（Asia/Shanghai）；分支 `dager-rank-and-experiment-records`，基于 `main` 的 `73ebac2`，尚未合并回 `main`。
- 用户已明确要求提交现有内容。2026-10-11 至 10-17 的代码按研究主题拆分提交，相关实验页已登记代码 commit，方法文档与聚合图同步整理；原始产物继续保存在本地 `outputs/`。
- `lm_prior_weight=0` 仍是默认值。提交不表示将先验、属性评分或生成候选切换为新的默认攻击；原版 `dager` 仍未实现。
- 本次全量 CPU 回归为 331 通过、3 跳过，Ruff 通过；实验索引测试 19 项通过，16 份实验记录及本地产物检查通过。本地与不含私有示例的干净检出均通过 MkDocs 严格构建。

接手顺序：`AGENTS.md` → 本文件 → `docs/experiments/README.md`。开始前检查 git、进程和 GPU，不能以调度器中的 `running` 字样判断任务存活。

## 本轮提交

| Commit | Content |
|---|---|
| `9efa65d` | 可选纯文本 LM prior、配对实验入口与测试 |
| `b304158` | 重构文本与 LM prior 文本的图像可区分性诊断 |
| `76e7c70` | 公开原型库属性推断、平衡扩展与对照 |
| `92169af` | 本地 Qwen 图像生成测试入口 |
| `561eeb2` | 合成 CT 候选的部位属性实验 |
| `a8282b3` | 混合模态头部属性实验、生成修复与测试 |

此前筛选研究 `7be400e`、图像排序诊断 `de95a9f`、rank 对比 `4bae2c8` 和日志修复 `e9a05eb` 已提交。历史展示材料位于 `docs/ppt/dager_branch_summary_20261011_v2/`，保留本地，不纳入 Git。

## 当前结论

- **文本恢复**：Residual 改善了第一阶段 token 排序，但候选完整不保证序列恢复。rank 对比见 [[dager-lora-rank]]，筛选研究见 [[dager-filter-phase]]。
- **纯文本先验**：rank 32 的冻结权重验证有收益；相同权重在 rank 8 收益有限，仍有候选漏检和早期剪枝。成功问答的重复度限制了结论，见 [[dager-lm-prior]]。
- **图像可区分性**：正确文本能支持部分全局排序，但局部像素优化仍失败；重构文本的内容与顺序都影响排序，LM prior 验证增益仍存在不确定性，见 [[gradient-discriminability]]、[[text-unknown-discriminability]]。
- **图像属性推断**：公开库三类平衡扩展显示相对 Model-only 的信号，但相对校准文本对照及同部位泛化的优势仍不明确，见 [[image-property-inference]]。
- **合成候选**：CT 部位诊断和混合模态头部扩展均已完成；历史样本重叠、患者相关性、构图差异及生成模态偏差仍限制结论，见 [[synthetic-ct-property]]、[[synthetic-head-property]]。
- **图像重构**：IG 仍未得到有效像素恢复；从近真值起点出发也可能漂移，不能把候选排序成功解释为图像重构成功，见 [[ig-failure-diagnosis]]。

## 下一步优先级

- **跨数据集与长度**：用户提出扩展到 VQA-RAD、VQAv2，并检验不同长度。先完成三个数据集的内容 token 长度、图像去重和问答重复度审计，再冻结分桶、抽样与对照。
- **实验入口**：`core/data.py` 已有两类新数据的导入器，`configs/data/` 已有对应配置；本地标准 manifest 目前只有 SLAKE。现有配对脚本写死了数据和模型路径，新实验需要最小参数化，不在换数据时同时改算法。
- **区分两个问题**：先用同一冻结模型捕获跨域客户端更新，再准备各数据集训练后的快照；两类结果分开报告。优先 rank 32，固定权重 0.01，与 Raw 和无先验 Residual 对照。
- **长度控制**：按实际内容 token 而非 padding 槽位计长，分别报告问题与答案。固定容量需预留 EOS；禁止静默截断后仍按原文长度解释结果。
- **文本方法**：保留现有 token filter。rank 8 优先诊断候选完整但真前缀被剪的样本的几何分差与 NLL 分差；rank 32 的最终排序失败单独保留，不用验证集反复调参。
- **属性方向**：若继续，先在独立开发数据上冻结评分与阈值，再核验同模态、跨部位的新目标；不要按本轮目标分数筛选生成候选或调整到全对。
- **其他待办**：TAG 的公开图像与问题条件仅有 pilot；IG 的旧参数扫描使用 bf16 候选，引用前需按当前 float32 主副本重跑。相关边界见 [[tag-text-only-public-image]]、[[ig-parameter-sweep]]。

## 运行状态与模型

- 本项目当前没有运行中的 GPU 实验。rank 8/32 先验、重构文本区分性、公开库属性、合成 CT 与混合模态头部实验均有完成标记，具体 worker 和失败记录保留在各实验页，不在本文件重复。
- 工作区盘点与 SLAKE 长度审计保存于 `outputs/diagnostics/workspace_audit_20261017/`。本轮只提交现有工作，没有启动跨数据集训练或攻击。
- 2026-10-17 检查时，PID 1151928 的 `train_stealth.py` 是其他任务；不要终止或修改。GPU 5 是共享 GPU，启动前重新核验空闲显存。
- rank 8 快照：`outputs/federation/slake_llava_f_l_fedavg/round-0010/`。
- rank 16、32 快照：`outputs/diagnostics/dager_rank_gpu5_20261010_v2/models/rank-16/round-0010/`、`rank-32/round-0010/`。
- GPU 环境为 Conda `gia`，CPU 测试为 `/tmp/giavlm-venv`；文档仅在本地查看。长任务必须 detached 启动并配 watcher，首个 capture/checkpoint 验证通过后才能离开。

## 复现与本地数据边界

- Observation 当前为 schema 5，模型 overlay 为 schema 4。DAGER checkpoint 为 schema 2；旧 checkpoint 只用于只读审计，不静默迁移续跑。
- rank 32 先验验证观测的轮次计数曾标为 0，但实际模型哈希对应第十轮；F-L 梯度逐项对照不受影响，原始观测未改写，后续捕获已修正，详见 [[dager-lm-prior]]。
- 私有图像、完整重构文本及样例图继续留在 `outputs/`。CT 讲解用的本地片段与符号链接位于 `docs/experiments/examples/`，该目录已忽略；页面通过可选片段保留本地展示，Git 中只保存说明与聚合结果。
- `GIA/`、其他参考项目、数据、模型和展示压缩包均按既有规则忽略。不要将生成图的模态提示直接当作真实模态标签。
- 本轮文档页将 `uncommitted` 标记改为 false，并记录相应代码 commit；这不意味着原始本地数据或大型产物被纳入 Git。
- 六组小型聚合结果保存在 `docs/experiments/data/` 并纳入 Git，包含对应本地来源的 SHA256；不包含逐样本文本、图像或原始样本标识。

---
重要提交、实验阶段结束或方向改变后更新本文件。具体数字与运行记录写入实验页，链接为 `[[id]]`；保持测量结果、假设与后续计划的区分。
