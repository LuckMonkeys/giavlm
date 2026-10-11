---
id: dager-lora-rank
title: LoRA rank 对 DAGER 文本重构的影响
date: 2026-10-10
track: text-dager
importance: key
data: slake/tune
model: llava-1.5-7b
tuning: F-L
round: 10
fed: fedsgd
knowledge: [image_known, private]
token_lengths_known: true
n: 10
evidence_level: development
outcome: mixed
uses_private_reference: false
contains_private_tokens: false
commits: [4bae2c8, e9a05eb]
code: [evaluation/dager_rank_study.py, tests/test_dager_rank_study.py, core/commands.py, tests/test_training_loss.py]
artifacts:
  - path: outputs/diagnostics/dager_rank_gpu5_20261010_v2/
    note: 用户明确授权后使用修复代码重新启动；两个模型及 80 次重构全部完成
  - path: outputs/diagnostics/dager_rank_gpu5_20261010_v2/comparison.json
    note: 三个 rank 的配对比较，原始图像与文本一致性检查通过
  - path: outputs/diagnostics/dager_rank_gpu5_20261010_v2/analysis.json
    note: 过滤、序列、子空间秩、配对差值和训练统计
  - path: outputs/diagnostics/dager_rank_gpu5_20261010_v2/recovery_audit.json
    note: 问答重复情况、答案长度分组及完整恢复的 token ID 核验
  - path: outputs/diagnostics/analysis_scripts/summarize_dager_rank_20261010.py
    note: 汇总与绘图脚本
  - path: outputs/diagnostics/dager_rank_gpu5_20261010_v2/first_checkpoint_audit.json
    note: 首个训练后快照校验通过，初始权重与第一次尝试一致，修复后的日志计算通过 GPU 验证
  - path: outputs/diagnostics/dager_rank_gpu5_20261010_v2/models_audit.json
    note: rank 16、32 均完成十轮训练，快照完整性及架构校验通过
  - path: outputs/diagnostics/dager_rank_gpu5_20261010_v2/rank16_execution_audit.json
    note: rank 16 的 40 次重构完成，来源代码和模型状态一致
  - path: outputs/diagnostics/dager_rank_gpu5_20261010/
    note: 冻结计划、训练配置、初始模型与失败记录；首轮日志计算 OOM 后停止
  - path: outputs/diagnostics/dager_rank_gpu5_20261010/failure_analysis.json
    note: OOM 发生位置、保存的状态和修复方案
  - path: outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100/summary.json
    note: 原 rank 8 的固定十样本基线
related: [dager-multisample-n10, dager-filter-phase]
---

## 要点

- **增大 rank 明显改善已知图像时的文本恢复**：Residual 条件的完整问答恢复从 rank 8 的 0/10，升到 rank 16 的 1/10 和 rank 32 的 5/10。
- **随机图像下仍未恢复完整问答**：rank 32 的问题更接近原文，答案恢复 3/10，但完整问题和完整问答仍为 0/10。
- **提升不只来自 token 候选覆盖**：rank 16 已知图像的全覆盖从 8/10 降到 7/10，问题恢复却从 4/10 升到 10/10；搜索实现和预算未变。
- **rank 32 的筛选接近完整覆盖**：Residual 在已知图像时覆盖全部样本，随机图像时覆盖 9/10；这是同一开发集上的配对结果，尚未独立验证。

## 关键数字

| Item | Count |
|---|---:|
| Paired images | 10 |
| New model ranks | 2 |
| Conditions per rank | 4 |
| Completed new reconstructions | 80 |
| Training rounds per new model | 10 |
| Unique QA sequences | 6 |

以下为 Residual 条件。Recall@100 按固定 100 个排名统计；Full coverage 使用实际 109 个候选，包括额外保留的 9 个 ambiguous ID。

| Rank | Image | Recall@100 | Full coverage | Question EM | Answer EM | Joint EM |
|---|---|---:|---:|---:|---:|---:|
| 8 | GT | 0.9795 | 8/10 | 4/10 | 0/10 | 0/10 |
| 16 | GT | 0.9786 | 7/10 | 10/10 | 1/10 | 1/10 |
| 32 | GT | 1.0000 | 10/10 | 10/10 | 5/10 | 5/10 |
| 8 | Random | 0.9580 | 6/10 | 0/10 | 0/10 | 0/10 |
| 16 | Random | 0.9786 | 7/10 | 0/10 | 0/10 | 0/10 |
| 32 | Random | 0.9938 | 9/10 | 0/10 | 3/10 | 0/10 |

## 研究问题

- 提高 LoRA rank 是否增加第一阶段可用的 token 筛选信号？
- 候选覆盖改善是否能转化为问题、答案或完整问答的恢复？
- rank 的影响是否依赖真实图像已知，或随机替代图像的条件？

## 实验设置

| Item | Setting |
|---|---|
| Model | LLaVA-1.5-7B，原固定 revision，bf16，model seed 17 |
| Training | F-L，FedAvg，AdamW，lr 0.0001，10 rounds |
| Client schedule | 1 client，batch 2，10 local steps，gradient accumulation 4，seed 42 |
| Snapshots | 每个 rank 保存初始模型与第 5、10 轮模型，原生最新状态按轮保存 |
| LoRA scaling | rank 8 的 alpha 为 16；rank 16 为 32；rank 32 为 64 |
| Initialization | 各 rank 使用同一原生初始化策略和种子；同一 rank 的所有条件共享模型快照 |
| Attack observation | 单样本 FedSGD 梯度，公开 token 长度 |
| Samples | 原开发集的 tune offsets 0–9，10 张不同图像 |
| Image knowledge | image_known；private 条件使用 seed 0 的随机替代图像 |
| Filter | 原始 Raw 与 Public residual，QKV 联合，informative K=100 |
| Sequence search | 原 dager_adapted，beam 16，prefix batch 32，200000 evaluations，3600 seconds，不做梯度重排 |
| Execution | 每个 rank 只加载一次攻击模型，逐次创建原有攻击器，不改变搜索实现 |
| Evaluation boundary | 两个新 rank 的全部重构提交后才读取参考评估；随后检查与 rank 8 样本一致性 |

## 结果

### 第一阶段：token 筛选

- Raw 的排名不依赖替代图像，两个图像条件的 token 指标相同；Recall@100 从 0.6094 升到 0.8120，再到 0.9358。
- Residual 在 rank 8 已接近全覆盖，rank 16 并没有在已知图像条件下继续改善；rank 32 才达到全部样本完整覆盖。
- rank 32 的随机图像条件仍有一个样本漏检，不能称作筛选问题已完全解决。
- 两层 QKV 子空间的数值秩分别为 24、48、96；数值秩增加并不意味着新增方向全是有效文本信号。

| Rank | Method | Image | Candidate recall | AP | Full coverage |
|---|---|---|---:|---:|---:|
| 8 | Raw | Both | 0.6094 | 0.2540 | 0/10 |
| 16 | Raw | Both | 0.8120 | 0.3381 | 2/10 |
| 32 | Raw | Both | 0.9358 | 0.4979 | 4/10 |
| 8 | Residual | GT | 0.9866 | 0.3526 | 8/10 |
| 16 | Residual | GT | 0.9786 | 0.3580 | 7/10 |
| 32 | Residual | GT | 1.0000 | 0.3718 | 10/10 |
| 8 | Residual | Random | 0.9580 | 0.3359 | 6/10 |
| 16 | Residual | Random | 0.9786 | 0.3563 | 7/10 |
| 32 | Residual | Random | 0.9938 | 0.3697 | 9/10 |

### 第二阶段：问题与答案恢复

- 已知图像时，rank 16 和 32 的 Residual 都完整恢复全部问题。rank 32 还完整恢复了五个答案，因此 Joint EM 为 5/10。
- rank 16 的候选覆盖没有改善，但问题恢复显著提高，说明变化不仅发生在第一阶段。不能把此前失败全部归因于固定 beam search 实现。
- 随机图像时，Residual 的问题 ROUGE-L 从 0.454 升到 0.825 和 0.898，WER 也下降；问题仍有错误，Question EM 保持为零。
- rank 32 已知图像时全部候选集都完整覆盖，但仍有五个答案未完整恢复，第二阶段仍有改进空间。

| Rank | Image | Question ROUGE-L | Question WER | Answer ROUGE-L | Answer WER |
|---|---|---:|---:|---:|---:|
| 8 | GT | 0.685 | 0.368 | 0.000 | 1.320 |
| 16 | GT | 1.000 | 0.000 | 0.286 | 0.930 |
| 32 | GT | 1.000 | 0.000 | 0.573 | 0.440 |
| 8 | Random | 0.454 | 0.680 | 0.000 | 1.340 |
| 16 | Random | 0.825 | 0.291 | 0.190 | 1.010 |
| 32 | Random | 0.898 | 0.204 | 0.410 | 0.590 |

[![不同 rank 的 Residual 筛选和文本恢复](figures/dager-lora-rank/comparison.png)](figures/dager-lora-rank/comparison.png)

### Raw 对照

- 不使用 Residual 时，提高 rank 也能改善已知图像下的问题与答案恢复。
- rank 32 的 Raw 在已知图像时恢复 3/10 个完整问答，低于 Residual 的 5/10。
- 随机图像下，Raw 在 rank 16 恢复过一个完整问题，在 rank 32 却没有，说明每项指标并非都单调改善。

| Rank | Image | Question EM | Answer EM | Joint EM |
|---|---|---:|---:|---:|
| 8 | GT | 2/10 | 0/10 | 0/10 |
| 16 | GT | 5/10 | 0/10 | 0/10 |
| 32 | GT | 8/10 | 4/10 | 3/10 |
| 8 | Random | 0/10 | 0/10 | 0/10 |
| 16 | Random | 1/10 | 0/10 | 0/10 |
| 32 | Random | 0/10 | 3/10 | 0/10 |

### 重复问答与答案长度审计

- 十张图像对应六种不同问答，重复次数为 3、2、2、1、1、1。不能把十张图像当成十种独立文本模式。
- rank 32 已知图像时恢复的五个完整问答覆盖三种不同问答，且通过了实际 token ID 完全一致的复核。
- 答案共有三个单 token 样本、七个多 token 样本。rank 32 的改善不只来自单 token 答案。

| Rank | Image | Single-token answer EM | Multi-token answer EM | Distinct recovered QA |
|---|---|---:|---:|---:|
| 16 | GT | 0/3 | 1/7 | 1 |
| 16 | Random | 0/3 | 0/7 | 0 |
| 32 | GT | 1/3 | 4/7 | 3 |
| 32 | Random | 1/3 | 2/7 | 0 |

### 首次运行与失败位置

- rank 16 的初始快照已经保存并通过哈希检查，包含 39,976,960 个可训练参数。
- 第一次 FedAvg 更新已执行，但随后计算记录用的 loss 时 OOM；最后提交的训练轮次仍为 0。
- 记录用前向将该轮的 80 个样本一次送入模型，而优化器更新使用 batch 2。
- 启动前 GPU 5 空闲约 50.6 GB；出错时又有一个其他进程占用约 17.7 GiB。分配请求为 4.46 GiB，当时只剩 3.04 GiB。
- 首次尝试中 rank 32 尚未启动，两个新 rank 都没有生成文本重构结果。该次失败不是重构方法的阴性结果。

### 已完成的修复

- `core.commands.selected_client_loss` 使用声明的训练 batch 大小做前向，按各块有效目标 token 数加权，包含 EOS。
- 该函数只用于训练后日志，不参与优化器或模型选择；训练数据、batch、local steps、累积次数和精度均未修改。
- bf16 下分块归约可能改变日志数值的微小舍入，不能要求与旧日志逐位一致。
- CPU 测试覆盖不同目标长度、空答案、最后一个不满块、参数不变，以及与整批计算的数值对照。
- 遵守仓库 `Never retry OOM` 的限制，首次失败后没有自动重试；用户随后明确回复“允许”，授权修复后的新实验。

### 授权后的新运行

- 新目录为 `outputs/diagnostics/dager_rank_gpu5_20261010_v2/`，旧产物完整保留。
- 修复后所有日志前向都使用原训练 batch 2，启动门槛为 32,000 MiB 空闲显存，低于门槛时等待；不结束其他任务。
- rank 16 和 rank 32 均已完成十轮训练，第十轮模型哈希和 rank、alpha 配置校验通过；80 次重构全部 completed。
- 与 rank 8 的实际图像和 token 张量逐样本一致性检查通过；两个新 rank 的全部重构提交后才运行评估。

## 结论

- 本轮 rank 32 在已知图像条件下取得明显更好的文本重构，原搜索实现无需修改也能恢复部分完整问答。
- 更大的 rank 同时改变梯度可见维度和训练后的模型状态。结果支持继续研究 rank 的影响，但尚不能分离这两个因素。
- 随机图像条件仍是主要困难：token 集合接近完整，部分答案可以恢复，问题序列也更接近原文，但完整问答仍为零。

## 局限

- 仅覆盖十张开发集图像、六种问答和一个训练种子。按图像 bootstrap 的区间不能代表独立问答模板上的泛化误差。
- 不同 rank 的参数形状不同；同一种子并不意味着不同 rank 的 LoRA 矩阵逐元素一致。
- 这是各 rank 经过相同训练预算后的模型比较，不能将差异全部归因于固定权重下梯度维数的变化。
- 测试使用改编版 DAGER 的固定 beam search，不代表原论文的完整实现。
- 使用共享 GPU，耗时不能直接解释成不同 rank 的固有速度差异。

## 下一步

- 在独立图像和更多不同问答上验证 rank 32，重点检查随机图像条件。
- 对 rank 32 已覆盖全部 token 但恢复失败的样本，诊断真实前缀在第二阶段的排名及淘汰位置。
- 对随机图像下高 ROUGE-L 但零 Question EM 的结果，区分少量 token 错误、位置偏差和上下文打分偏差。

??? note "运行记录"

    - 使用用户授权的 GPU 5。
    - detached worker PID 743410，watcher PID 743411 均已退出，显存已释放；失败状态保存在 failed.json。
    - 授权后的 worker PID 745473、watcher PID 745474 均已正常退出，完成标记与汇总哈希匹配；总耗时 6310.4 秒，约 105.2 分钟。
    - 两项新 CPU 测试通过，验证训练配置不变与公开输入封装和原攻击结果一致；Ruff 通过。
    - 测试初稿误用了 token 产物文件名，已更正并通过重测；没有因此启动失败的 GPU 实验。
    - 日志修复及相关工作流的 38 项测试通过；修复后全量 CPU 回归 293 通过、3 跳过，Ruff 通过。
    - 11 条实验记录通过索引和产物检查，MkDocs 本地构建通过；修复后的 GPU 路径已通过首轮训练验证。
