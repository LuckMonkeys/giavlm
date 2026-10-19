# 项目交接

- 更新时间：2026-10-18（Asia/Shanghai）；分支 `dager-rank-and-experiment-records`，基于 `main` 的 `73ebac2`，最新 commit `cedd460`。工作区干净，分支尚未推送，也未合并回 `main`。
- 用户已明确要求提交现有内容。2026-10-11 至 10-18 的代码按研究主题拆分提交，相关实验页已登记代码 commit，方法文档与聚合图同步整理；原始产物继续保存在本地 `outputs/`。
- `lm_prior_weight=0` 仍是默认值。提交不表示将先验、属性评分或生成候选切换为新的默认攻击；原版 `dager` 仍未实现。
- 本轮五个新 commit 均在干净检出中逐一验证：全量 CPU 回归最终为 355 通过、4 跳过，Ruff 通过；文档相关测试 36 项通过。14 份实验记录及本地产物检查通过，MkDocs 严格构建通过。

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
| `2440189` | Qwen3.5 adapter 与 DAGER 线性注意力输入 |
| `6beafa7` | 跨模型文本恢复 benchmark 与只读巡检 |
| `e2ce836` | 通用摄影候选库的头部属性对照 |
| `eb0c5a4` | 仅限 loopback 的文档在线编辑器、首页与索引卡片、中文搜索 |
| `cedd460` | 合并实验记录、记录结构校验、benchmark 实验页与 WN-A100 规则 |

此前筛选研究 `7be400e`、图像排序诊断 `de95a9f`、rank 对比 `4bae2c8` 和日志修复 `e9a05eb` 已提交。`cedd460` 将 `dager-multisample-n10`、`dager-lora-rank` 合并为 [[dager-multisample]]，将 Qwen 图像生成、合成 CT 与头部属性三页合并为 [[synthetic-property]]；旧 id 已不可链接。历史展示材料位于 `docs/ppt/dager_branch_summary_20261011_v2/`，保留本地，不纳入 Git。

## 当前结论

- **文本恢复**：Residual 改善了第一阶段 token 排序，但候选完整不保证序列恢复。rank 对比见 [[dager-multisample]]，筛选研究见 [[dager-filter-phase]]。
- **纯文本先验**：rank 32 的冻结权重验证有收益；相同权重在 rank 8 收益有限，仍有候选漏检和早期剪枝。成功问答的重复度限制了结论，见 [[dager-lm-prior]]。
- **图像可区分性**：正确文本能支持部分全局排序，但局部像素优化仍失败；重构文本的内容与顺序都影响排序，LM prior 验证增益仍存在不确定性，见 [[gradient-discriminability]]、[[text-unknown-discriminability]]。
- **图像属性推断**：公开库三类平衡扩展显示相对 Model-only 的信号，但相对校准文本对照及同部位泛化的优势仍不明确，见 [[image-property-inference]]。
- **合成候选**：医学外观候选支持头部属性诊断，但三套通用人体摄影库未保留该效果，正确文本也未恢复表现。历史样本重叠、患者相关性、候选外观差异及生成模态偏差仍限制结论，见 [[synthetic-property]]。
- **图像重构**：IG 仍未得到有效像素恢复；从近真值起点出发也可能漂移，不能把候选排序成功解释为图像重构成功，见 [[ig-failure-diagnosis]]。

## 下一步优先级

- **已完成的执行计划**：[跨模型文本恢复 benchmark](plans/2026-10-17-crossmodel-text-benchmark.md) 的检查清单全部完成。结果与表图见 [[crossmodel-text-benchmark]]；不自动开启额外实验。
- **跨数据集与长度**：三个数据集的冻结分桶、图像隔离、文本重叠与原生长度均已审计，72 个 QA 的配对矩阵已完成。两个新数据集的长桶各 n=1，不能据此声明稳定趋势。
- **实验入口**：`evaluation/text_benchmark_data.py` 准备与冻结数据，`evaluation/text_benchmark.py` 执行配对捕获与攻击，`evaluation/text_benchmark_report.py` 生成聚合表图；Qwen 使用独立 `gia-qwen35` 环境，旧 `gia` 保留。
- **本轮训练口径**：每个模型与数据集组合独立训练到 Round 10，再单独捕获单样本 FedSGD 更新。主实验使用 rank 32、alpha 64 与 LM prior 权重 0.01；跨域冻结模型和更多方法消融不属于本轮必需矩阵。
- **长度控制**：按实际内容 token 而非 padding 槽位计长，分别报告问题与答案。固定容量需预留 EOS；禁止静默截断后仍按原文长度解释结果。
- **文本方法**：保留现有 token filter。rank 8 优先诊断候选完整但真前缀被剪的样本的几何分差与 NLL 分差；rank 32 的最终排序失败单独保留，不用验证集反复调参。
- **属性方向**：通用摄影库对照已完成，先检查候选领域与外观差异，再决定是否利用可恢复的文本线索生成领域候选；见 [[synthetic-property]]。后续仍应在独立开发数据上冻结评分与阈值，不按当前目标分数筛选候选或调整到全对。
- **其他待办**：TAG 的公开图像与问题条件仅有 pilot；IG 的旧参数扫描使用 bf16 候选，引用前需按当前 float32 主副本重跑。相关边界见 [[tag-text-only-public-image]]、[[ig-parameter-sweep]]。

## 运行状态与模型

- 2026-10-18 通用摄影候选库对照已完成，见 [[synthetic-property]]。用户明确允许共享本机 GPU 5，UUID `GPU-8ad9e9b4-e159-4f78-7036-2d1d018e7b7d`。根目录 `outputs/diagnostics/generic_head_n20_gpu5_20261018/`，`generation-source`、`runtime-source` 固定代码；复用原二十张目标的观测与重构文本，未重新训练或搜索文本。生成 worker 2123789、watcher 2123790 和评估 worker 2142475、watcher 2142476 均已退出；七十二张生成图全部保留，120 份预测、输入与模型一致性校验完成，`complete.json` 和 `report_complete.json` 均已保存。本次 GPU 已释放，不要重复启动；评估代码已提交为 `e2ce836`，原医学库结果未覆盖。
- 用户已批准跨模型、跨数据集实验，完整步骤、固定参数与恢复规则见上述计划。本项目全部任务合计最多同时占用 WN-A100 的两张空闲物理卡，GPU 编号与 UUID 启动前重新核验。
- WN-A100 已通过 SSH 别名 `zx-wn-a100` 验证可达，具备 8 张 A100 80GB；共享仓库、`gia` Python 和 Qwen3.5-4B 权重均可访问。后续可以从当前会话经 SSH 执行实验，连接与运行规则已写入 `AGENTS.md`；启动前重新检查远端 GPU 占用与运行时兼容性。
- 本轮 benchmark 已完成，完整结果与边界见 [[crossmodel-text-benchmark]]。六个组合的十轮训练及 288 次配对攻击均完成，全部攻击为 completed；六组 summary、报告来源 SHA256 与 ZIP CRC 已核验。原 `gia` 未变更，代码和新环境均在本机准备，GPU 程序只在 WN-A100 两张卡上执行。代码已提交为 `2440189`、`6beafa7`。
- CPU 阶段调度器原 PID 835、watcher 836，`controller.json` 为 completed、`all_gpu_jobs_complete`；后续不要根据历史启动记录重复运行已完成任务。
- 本机 CPU 汇总原 PID 1507970，`finalizer.json` 为 completed。完整表图位于本轮根目录 `report/`，打包为 `report-aggregate.zip`，含 18 个聚合文件，无私有示例；文档内聚合副本与 SHA256 清单位于 `docs/experiments/data/crossmodel-text-benchmark/`，四张完整矩阵图已更新。`interim/slake/` 仅为历史阶段输出。
- 用户要求的每 10 分钟后台巡检已正常退出，原 PID 1525580；`monitor/timer.json` 为 finished，完成 31 次检查，最终 health 为 completed。最新报告为 `monitor/latest.md`，历史为 `monitor/history.jsonl`；没有下一次检查，不需要重启已完成实验的监控。
- 用户要求的会话主动监控也已执行至实验完成；最后巡检为 2026-10-18 03:18，`monitor/live-session.json` 记录 completed、288/288。监控期间未自动重试 OOM、改预算或改实验条件。
- 工作区盘点与 SLAKE 初步审计保存于 `outputs/diagnostics/workspace_audit_20261017/`。Q64、A64 容量变化使旧 SLAKE 快照不再直接复用，本轮六个组合均需重新训练。
- `gia-qwen35` 已在本机创建于共享 NAS，安装 Transformers 5.20.1、PEFT 0.21.2、Accelerate 1.15.0，保留 Torch 2.6.0；新环境 `pip check` 通过，旧 `gia` 的 pip 快照前后一致。本轮产物根目录为 `outputs/diagnostics/crossmodel_text_20261017/`；训练使用冻结 `source-smoke-v1`，正式攻击使用 `source-benchmark-v1`。数据准备完成：`data/*/frozen.json` 共冻结 72 个 QA，正式配对矩阵为 288 次攻击；VQAv2 固定训练与评估调度使用的 816 张图片已校验。
- 2026-10-18 检查时，本机没有本项目的运行进程；此前其他任务 PID 1151928 已不在进程表中。
- rank 8 快照：`outputs/federation/slake_llava_f_l_fedavg/round-0010/`。
- rank 16、32 快照：`outputs/diagnostics/dager_rank_gpu5_20261010_v2/models/rank-16/round-0010/`、`rank-32/round-0010/`。
- GPU 环境为 Conda `gia`，CPU 测试为 `/tmp/giavlm-venv`；文档仅在本地查看。长任务必须 detached 启动并配 watcher，首个 capture/checkpoint 验证通过后才能离开。

## 复现与本地数据边界

- Observation 当前为 schema 5，模型 overlay 为 schema 4。DAGER checkpoint 为 schema 2；旧 checkpoint 只用于只读审计，不静默迁移续跑。
- rank 32 先验验证观测的轮次计数曾标为 0，但实际模型哈希对应第十轮；F-L 梯度逐项对照不受影响，原始观测未改写，后续捕获已修正，详见 [[dager-lm-prior]]。
- 私有图像、完整重构文本及样例图继续留在 `outputs/`。CT 讲解与用户要求的头部属性六图对照，通过 `docs/experiments/examples/` 下的本地片段和符号链接展示，该目录已忽略；页面使用可选片段，Git 中只保存说明与聚合结果。六图对照见 [[synthetic-property]] 的图像直观对比，真图及两类候选按固定记录顺序选择，未按推断结果挑选。
- `GIA/`、其他参考项目、数据、模型和展示压缩包均按既有规则忽略。不要将生成图的模态提示直接当作真实模态标签。
- 本轮文档页将 `uncommitted` 标记改为 false，并记录相应代码 commit；这不意味着原始本地数据或大型产物被纳入 Git。
- 小型聚合结果按实验保存在 `docs/experiments/data/` 并纳入 Git，包含对应本地来源的 SHA256；不包含逐样本文本、图像或原始样本标识。

---
重要提交、实验阶段结束或方向改变后更新本文件。具体数字与运行记录写入实验页，链接为 `[[id]]`；保持测量结果、假设与后续计划的区分。
