---
id: dager-single-sample-diagnostics
title: DAGER 单样本诊断
date: 2026-10-02
track: text-dager
importance: archive
data: slake/tune
model: llava-1.5-7b
tuning: F-L
round: 10
fed: fedsgd
knowledge: [image_question_known, image_known]
token_lengths_known: true
n: 1
evidence_level: diagnostic
outcome: mixed
uses_private_reference: true
private_reference_note: 仅“替代图敏感性”一节把真实图像和随机像素混合
contains_private_tokens: false
superseded_by: dager-multisample
archive_note: >-
  只有一个样本、一个梯度；residual 和 random surrogate 的主要结论已在 10 个样本上复测。
  保留这页是因为空白对照、QKV 消融和混合图像分析没有在多样本上重复。
commits:
  - 6bcdc6c
  - 349ab9f
  - 0a61e94
  - 03f214f
  - 86aefe7
  - 6a7de71
  - 0f64c28
  - 187da5c
  - 691580e
code:
  - attacks/analytic
  - evaluation/dager_projection_ablation.py
  - evaluation/dager_surrogate_diagnostic.py
artifacts:
  - path: outputs/diagnostics/dager_real_gpu4_f_l_r10/run-00000/
    note: image_question_known（只恢复答案的上界）
  - path: outputs/diagnostics/dager_real_gpu4_f_l_r10_image_known/run-00000/
    note: image_known + token 长度公开（目标条件）
  - path: outputs/diagnostics/dager_projection_ablation_valid_vocab/
    note: 有效的配对比较和报告
  - path: outputs/diagnostics/dager_projection_ablation/
    note: 无效（包含 padding 的 embedding 行），只作审计记录，不能引用
  - path: outputs/diagnostics/dager_projection_ablation_valid_vocab/filter_validation.json
    note: 汇总报告；不保留任何错误参考文本或其 token ID
  - path: outputs/diagnostics/dager_surrogate_gpu5_n1/
    note: 27 个打分产物和评估结果
  - path: outputs/diagnostics/dager_projection_qkv_gpu5_n1/
    note: summary.json, top50_comparison.json, ranking_validation.json
  - path: outputs/diagnostics/analysis_scripts/run_dager_projection_gpu5.sh
  - path: outputs/diagnostics/analysis_scripts/watch_dager_projection_gpu5.sh
related: [dager-multisample]
---

## 要点

- **residual 明显优于 raw**：在这个样本上，Top-50 命中 13/14，raw 只有 7/14。
- **信号远高于随机，但问题 token 的命中多半来自词频**：按语料词频取 Top-50 就能覆盖全部 10 个问题 token，但覆盖不到任何答案 token。
- **视觉编码器生成的子空间都有帮助**：纯随机像素图也能达到 10/14，真实图像为 13/14。
- **联合 QKV 的打分最好**：QKV 为 13/14，单个投影只有 7–9/14。
- **所有句子搜索都失败**：每次运行的问题和答案 EM 都是 0。

## 分析概览

这 5 项分析用的是**同一个**样本和同一个梯度，都在 2026-10-02 完成，早于多样本研究。

| Analysis | 发现 | 在 [[dager-multisample]] 中的情况 |
|---|---|---|
| 第一次真实诊断 | token 检测准确但不完整；句子 EM 为 0 | 已复现：句子恢复仍然失败 |
| Raw vs residual | Top-50 recall 从 7/14 升到 13/14 | 已在 10 个样本上复现（K=100 时 0.61 → 0.98） |
| 空白对照 | 远高于随机；问题 token 的命中多半来自词频 | 没有逐样本重复，仍是主要的保留意见 |
| 替代图敏感性（使用私有数据） | 随机像素图 10/14，真实图像 13/14 | 已复现：random surrogate 保留了大部分收益 |
| Q/K/V 消融 | 联合 QKV 13/14，单个投影 7–9/14 | 没有重复；之后一律使用 QKV |

## 研究问题

- **第一次诊断**：`dager_adapted` 在真实训练后的 LLaVA 更新上，能否检测出 token 集合或恢复出句子？
- **Raw 与 Residual**：residual 打分比 raw 好多少？这个优势是不是只来自模板和词频？
- **替代图**：residual 的收益有多少依赖真实图像，多少来自视觉编码器的通用子空间？
- **Q/K/V 投影**：单个 LoRA 投影和联合 QKV 打分是否等价？

## 实验设置

| Item | Setting |
|---|---|
| 性质 | 开发诊断（n=1），不是论文结果，也不用于调阈值<br>"替代图敏感性"一节用真实图像构造混合图，属于使用私有数据的诊断；其余各节只使用公开输入 |
| 样本 | 1 个经授权的 SLAKE `tune` 开发样本 |
| 受害模型 | LLaVA-1.5-7B，F-L 第 10 轮快照，LoRA rank 8，bf16 |
| 联邦设置 | FedSGD |
| 攻击者知识 | 知道答案的 token 长度<br>`image_question_known`：知道图像和问题，相当于只恢复答案的上界<br>`image_known`：只知道图像和长度（问题 10 个内容 token，答案 4 个），这是目标条件 |
| 默认攻击配置 | residual 打分，联合 QKV，阈值 0.05 |
| 防泄露措施 | 每次攻击先提交完整的词表打分，评估才读取私有 token ID |

## 结果

### 第一次诊断

**回答**：informative 候选全部正确但覆盖不全；两种知识条件下句子 EM 都是 0。

- 扫描了全部 32,064 个词表 ID。
- 两种知识条件下，informative 候选都全部正确（precision 1.0），但数量太少，覆盖不全。
- 问题私有时，token 覆盖明显变差：只找到 14 个私有 token 中的 1 个。
- 两种条件下，beam search 都填满了所有私有位置，但问题和答案的 EM、ROUGE、word recall 全部为 0。

| Knowledge | Informative | Ambiguous | Precision (informative) | Recall | Question recall | Answer recall | EM |
|---|---:|---:|---:|---:|---:|---:|---:|
| image_question_known | 2 | 19 | 1.0 | 0.5（2/4） | – | 0.5 | 0 |
| image_known | 1 | 9 | 1.0 | 0.071（1/14） | 0 | 0.25 | 0 |

- `image_question_known` 下，Recall 的分母是 4 个答案 token；加上 ambiguous 候选后 precision 为 0.095，recall 仍为 0.5。
- `image_question_known` 下，第一、二层商空间的秩都是 24（隐藏维度 4096 中，公开子空间的秩分别为 597 和 653），两个空间都没有饱和。
- 搜索开销很小：`image_question_known` 用 1,029 次前缀评估（65 个 batch），模型加载后 5.5 秒；`image_known` 用 2,030 次（128 个 batch），6.1 秒。都没有做全梯度重排。

### Raw 与 Residual

**回答**：Residual 的 Top-50 命中从 7/14 升到 13/14，远高于随机；问题 token 的命中多半来自词频，答案 token 才是样本信号。

- residual 的 Top-50 命中从 7/14 升到 13/14，问题和答案的召回都明显提高。
- 两个 Top-50 集合只有 22 个 ID 重合（Jaccard 0.282），说明去除公开方向大幅改变了词表排序。
- residual 的 Top-50 中有 9 个 ambiguous ID；唯一漏掉的 token 排在第 98 位。
- 这一节复用了上一节 `image_known` 的公开观测，没有重新捕获；句子搜索只做了一个前缀，不参与比较。

| Method | Top-50 precision | Recall | Question recall | Answer recall | True-token rank range | Median rank |
|---|---:|---:|---:|---:|---|---:|
| Raw | 0.14 | 0.50（7/14） | 0.40 | 0.75 | 2–26,249 | 389.5 |
| Residual | 0.26 | 0.929（13/14） | 0.90 | 1.0 | 10–98 | 16.5 |

#### 空白对照

- **远高于均匀随机**：随机取 Top-50 时，期望只命中 14 个私有 ID 中的 0.0219 个。
    - raw 至少命中 7 个的精确概率为 4.99e-17，residual 至少命中 13 个的概率为 8.41e-37。
    - 100,000 次随机抽样中，没有一次达到这两个结果。
- **问题 token 的命中多半来自词频**：按语料词频取 Top-50，能覆盖全部 10 个问题 token，但覆盖不到任何答案 token。
- **答案 token 是真正的样本信号**：residual 仍然找到了全部 4 个答案 token，而这些 token 不在词频 Top-50 中。
- **错误文本对照**（共 632 条，来自其他图像组的 SLAKE `tune` 文本；比较时排除 token 集合完全相同的文本）：
    - residual 的问题召回为 0.90，错误文本的中位数约为 0.46（593 条文本，plus-one 经验尾概率 0.00168）。
    - residual 的答案召回为 1.0，错误文本的中位数为 0（619 条文本，尾概率 0.0306）。
    - raw 的问题召回 0.40 不算突出（尾概率 0.180），但答案召回 0.75 是突出的（尾概率 0.0226）。
- 长度也匹配的错误文本，去重后每个字段只剩 6 条，最小可达尾概率为 1/7，无法证明显著性。描述性地看，这些文本的答案重合对两种打分都为 0。

不同 K 下的命中数：

| Method | K=20 | K=50 | K=100 | K=1000 | AP |
|---|---:|---:|---:|---:|---:|
| Residual | 9/14 | 13/14 | 14/14 | – | 0.336 |
| Raw | 5/14 | 7/14 | 7/14 | 8/14 | 0.187 |

### 替代图

**回答**：纯随机像素图已能达到 10/14，真实图像为 13/14；视觉编码器的通用子空间贡献了大部分收益。

- **任何来自视觉编码器的子空间都比 raw 好**：纯随机像素图也能达到 10/14，比 raw 多 3 个，比同秩的随机子空间多 2 个。
- **真实图像额外贡献 3 个命中**，其中包括第 4 个答案 token。
- **这个优势很脆弱**：只混入 1% 的随机像素（与真实图像仍相差 48.5 dB），命中就从 13/14 降到 11/14。
- 原因是视觉编码器对像素很敏感：1% 混合图的视觉特征余弦相似度只有约 0.914，视觉子空间重合度约 0.690。

| Image subspace | Top-50 hits | Question hits | Answer hits | AP |
|---|---:|---:|---:|---:|
| None (raw) | 7/14 | 4/10 | 3/4 | 0.187 |
| Template only | 8/14 | – | – | 0.183 |
| Matched random subspace | 8/14 | – | – | ≈0.255 |
| 25–100% random-pixel blend | 10/14 | 7/10 | 3/4 | ≈0.256–0.262 |
| 10% random-pixel blend | 10–11/14（平均 10.67） | – | – | – |
| 3% random-pixel blend | 11/14 | – | – | – |
| 1% random-pixel blend | 11/14 | 8/10 | 3/4 | 0.287 |
| True image | 13/14 | 9/10 | 4/4 | 0.336 |

- 模板子空间的秩为 11，视觉部分增加的秩为 576，所有完整视觉条件的公开秩都是 587；所以同秩随机子空间去掉的维度数完全相同。
- 所有条件下都没有真实 token 被标为 ambiguous。
- 21 个混合图中，视觉子空间重合度与 Recall@50 的 Spearman 为 0.826，与 AP 为 0.917。这是同一个梯度的重复测量，只作描述，不作推断。
- 纯随机图像的子空间重合度仍有约 0.544，特征余弦相似度约 0.727，说明视觉编码器会把无关像素映射到一个共享的视觉流形上。
- 候选集大小的影响：一张典型的纯随机图像在 K=20、50、100、200、500 时分别命中 9、10、11、13、14 个；真实图像在 K=100 时就达到 14/14；raw 到 K=1000 也只有 8/14。
- 这一节在 GPU 打分阶段从不读取私有文本；27 个打分产物全部提交后，评估才对齐 token ID。

### Q/K/V 投影

**回答**：不等价。联合 QKV 的 Top-50 命中为 13/14，单个投影只有 7–9/14。

- **联合 QKV 的排序最好**：Top-50 命中 13/14，单个投影只有 7–9/14。
- **单个投影之间没有稳定的优劣**：K=50 时 K 优于 V，K=100 时 V 优于 K。
- **四种投影的句子搜索都失败**：问题和答案的 EM、ROUGE、word recall 都是 0。
- 在冻结的阈值下，Q 和 QKV 各保留 1 个正确的 informative token 和 9 个 ambiguous ID；K 和 V 只保留 9 个 ambiguous ID（都不是私有 token）。对应的候选 recall 分别为 1/14、0/14、0/14、1/14。
- QKV 保存的打分向量和 ambiguous 标记，与之前 residual 的产物完全一致（最大差值为 0），说明配对比较保留了原来的打分基线。

| Projection | Layer 1 rank | Layer 2 rank | Hits@20 | Hits@50 | Hits@100 | Question recall@50 | Answer recall@50 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q | 8 | 8 | 4/14 | 7/14 | 7/14 | 0.40 | 0.75 | 0.1209 |
| K | 8 | 8 | 8/14 | 9/14 | 9/14 | 0.60 | 0.75 | 0.2088 |
| V | 8 | 8 | 7/14 | 8/14 | 11/14 | 0.50 | 0.75 | 0.2051 |
| QKV | 24 | 24 | 9/14 | 13/14 | 14/14 | 0.90 | 1.00 | 0.3355 |

- 设置：对现有的 `image_known` 公开观测分别跑 4 次攻击（`projections=q/k/v/qkv`），应用户要求进行。
- 没有重新捕获，没有换图像，没有调阈值，也没有改受害模型。其余设置沿用原协议：阈值 0.05，beam 16，最多 10,000 次评估，3,600 秒，不做梯度重排。
- 每次运行在模型加载后约 6.8–7.3 秒；Q 和 QKV 用 2,030 次前缀检查，K 和 V 用 1,818 次。

## 结论

- 在这个样本上，DAGER 打分包含真实的、与样本相关的 token 信号，答案部分尤其明显。
- 去除公开方向能增强这个信号。
- 打分也明显偏向常见词和模板词，所以 13/14 不能当作 13 个与样本相关的发现。
- raw 只是一个"不用图像"的操作性基线，不是理论下界；错误的残差空间理论上也可能损害排序。
- 联合 QKV 在低秩 LoRA 设置下优于单个投影。
- 句子恢复全部失败。

## 局限

- 只有一个样本、一个梯度，结果都是事后分析，不能推广。
- "替代图敏感性"用真实图像构造混合图，只能用于诊断，不是图像私有条件下的攻击。
- 等 K 的指标都是在结果提交后计算的；完整搜索仍使用阈值 0.05，评估后没有调整任何阈值。

## 下一步

- 主要结论已在 [[dager-multisample]] 中用 10 个样本复测。
- 空白对照和 QKV 消融尚未在多样本上重复。

??? note "运行记录"
    - 第一次运行时捕获成功，但暴露了一个新运行与恢复的接口问题：`ExperimentRunner` 向新攻击传入了 `resume=true`。
    - commit `0a61e94` 让 DAGER 只在自己的检查点存在时才恢复，并增加了回归测试。
    - 之后直接对未改动、经哈希校验的公开观测运行攻击，结果提交后再评估；重试没有改变数据、精度、训练条件或搜索设置。
    - 攻击目录中残留的 `failed.json` 记录的是搜索前的接口失败；`result.json` 和 `evaluation.json` 是之后成功运行的产物。
    - 最初的 raw 比较是无效的：Top-50 中有 49 个是超出 tokenizer 词表的 padding embedding 行（LLaVA 有 32,064 个 embedding 行，tokenizer 只有 32,002 个 ID）。commit `03f214f` 排除了这些行，并增加了回归测试和等 K 评估器。无效产物保留在 `outputs/diagnostics/dager_projection_ablation/`，只作审计记录，不能引用。
    - GPU 4 运行前为空，运行后回到 0 MiB；没有 OOM，也没有在读取参考后调整阈值。
    - 替代图实验和 QKV 消融在 GPU 5 上运行，与无关任务共享（已占用约 40.5–40.6 GiB）；替代图实验用了约 16.3 GiB，均无 OOM，显存已释放。
    - DAGER 相关的 CPU 测试通过（24 个），新 runner 的 Ruff 检查通过。
