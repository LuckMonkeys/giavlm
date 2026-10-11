---
id: dager-multisample-n10
title: DAGER 多样本文本重建（n=10）
date: 2026-10-03
end_date: 2026-10-05
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
contains_private_tokens: true
commits: [388241f, 887e87e, 7edac7f, f9d9fd3, cf402bd]
code: [evaluation/dager_reconstruction_study.py]
artifacts:
  - path: outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100/
    note: plan.json, configs/, summary.json, complete.json, run-*/
related: [dager-single-sample-diagnostics]
---

## 要点

- **私有文本中的大部分token能正确过滤**： 当真实图像已知时(GT), 使用residual方法减去公开方向，Recall@100 为 0.98；当真实图像未知时，使用随机图像代替，Recall@100 仍有 0.96。
- **梯度空间中减去公开方向效果明显**：使用原始DAGER算法token过滤， Recall@100 只有 0.61。
- **rank 8 的完整文本恢复失败**：本页十个样本中，没有一个完整恢复出问题和答案；后续提高 rank 的结果见 [[dager-lora-rank]]，不能把本页结论推广到所有 rank。

## 关键数字

| Method | Image | Recall@100 | Full coverage (actual) | Question EM | Answer EM | Joint EM |
|---|---|---:|---:|---:|---:|---:|
| Residual | GT | **0.98** | 8/10 | 4/10 | **0/10** | **0/10** |
| Residual | Random | **0.96** | 6/10 | 0/10 | **0/10** | **0/10** |
| Raw | GT | 0.61 | 0/10 | 2/10 | **0/10** | **0/10** |
| Raw | Random | 0.61 | 0/10 | 0/10 | **0/10** | **0/10** |

## 研究问题

- residual 打分的 token 筛选效果，能否从单个样本推广到多个样本？
- 图像私有时，random surrogate 还有没有用？
- token 覆盖得好，能否进一步恢复出句子？

## 实验设置

| Item | Setting |
|---|---|
| 模型 | LLaVA-1.5-7B，F-L 第 10 轮快照，LoRA rank 8，bf16 |
| 联邦设置 | FedSGD |
| 样本 | SLAKE `tune` 集中 10 个不同的图像组（offset 0–9）<br>是 10 个不同样本，不是同一个样本换 10 个随机种子 |
| 攻击者知识 | 知道问题和答案的 token 长度<br>不知道问题和答案的内容 |
| 图像条件 | 攻击者知道真实图像<br>或者图像私有，攻击者用自己生成的随机像素图（random surrogate）代替<br>random surrogate 由攻击种子 0 生成，没有针对私有数据优化或挑选 |
| 打分方式 | Residual：先去除公开方向再打分<br>Raw：直接打分 |
| 条件命名 | `Method`：Residual / Raw（方法变体）<br>`Image`：GT（真实图像）/ Random（random surrogate）（实验条件）<br>两者组合成 4 个条件，正文写作 Residual (GT)、Raw (Random) 等 |
| 候选数量 | 取 100 个 informative 候选<br>residual 条件另外保留 9 个 ambiguous ID，实际共 109 个<br>raw 条件正好 100 个 |
| 搜索设置 | 联合 QKV，beam 16<br>最多 200,000 次前缀评估，限时 3,600 秒<br>不做全梯度重排 |
| 防泄露措施 | 图像私有的条件下，没有任何公开图像产物<br>同一样本在两种图像条件下上传的更新，哈希值完全一致<br>40 次攻击全部提交之后，评估才读取私有参考 |

## 结果

### Token 筛选

- residual 在全部 10 个配对样本上都优于 raw。
- 真实图像只在 2 个样本上优于 random surrogate，其余 8 个持平，两者的平均 AP 只差 0.0167。
- raw 打分不使用图像信息，所以两种图像条件下的 raw 排名完全相同。

指标是每个样本"问题和答案去重 token"上的宏平均：

| Method | Image | Actual candidates | Candidate recall | Recall@100 | AP | Full coverage (actual) | Full coverage (K=100) |
|---|---|---:|---:|---:|---:|---:|---:|
| Residual | GT | 109 | 0.9866 | 0.9795 | 0.3526 | 8/10 | 7/10 |
| Residual | Random | 109 | 0.9580 | 0.9580 | 0.3359 | 6/10 | 6/10 |
| Raw | GT | 100 | 0.6094 | 0.6094 | 0.2540 | 0/10 | 0/10 |
| Raw | Random | 100 | 0.6094 | 0.6094 | 0.2540 | 0/10 | 0/10 |

### 问题重建

- 有真实图像时，residual 能完整恢复 4 个问题。
- 只有 random surrogate 时，一个问题也恢复不出来。

| Method | Image | EM | ROUGE-1 | ROUGE-2 | ROUGE-L | Word recall | WER |
|---|---|---:|---:|---:|---:|---:|---:|
| Residual | GT | 0.400 | 0.700 | 0.516 | 0.685 | 0.670 | 0.368 |
| Residual | Random | 0.000 | 0.467 | 0.264 | 0.454 | 0.471 | 0.680 |
| Raw | GT | 0.200 | 0.323 | 0.240 | 0.323 | 0.295 | 0.705 |
| Raw | Random | 0.000 | 0.299 | 0.210 | 0.299 | 0.295 | 0.768 |

### 答案重建

- 四种条件下，答案的 EM、normalized EM、ROUGE-1/2/L 和 word recall 全部为 0。
- 答案的平均 WER 为 1.32–1.34。
- 问题和答案同时完整恢复的样本：所有条件下都是 0/10。

### 漏检分析（2026-10-05 事后审计）

这是对已提交打分结果的事后分析，没有重跑实验，也不能当作选 K 的规则。

- 漏掉的大多是答案的最后一个子词。
- 增大 K 能补回大部分漏检。
- 但对简单样本，固定 K=100 会引入很多无关候选。

统计说明：先在排名中排除 ambiguous token，每个 Top-K 集合再额外加上 9 个 ambiguous 候选。
这和之前报告里"在全部允许词表上取固定 K"的口径不同。

不同 K 下的 full coverage 样本数：

| Image | K=20 | K=100 | K=200 | K=500 | K=1000 |
|---|---:|---:|---:|---:|---:|
| GT | 6/10 | 8/10 | 9/10 | 9/10 | 9/10 |
| Random | 5/10 | 6/10 | 7/10 | 9/10 | 9/10 |

漏检明细（JSON 中的 token 片段保留了前导空格）：

| Image | Run | Missed token (rank) |
|---|---|---|
| GT | 2 | 答案末尾 `ay`（135） |
| GT | 9 | 答案末尾 `ine`（1090） |
| Random | 0 | `ay`、`image`、`take`（169 / 243 / 319） |
| Random | 2 | `ay`（253） |
| Random | 4 | `ay`（182） |
| Random | 9 | `ine`（1441） |

- 简单样本的例子：Run 5 和 Run 6 在两种图像条件下，6 个真实 token 都排在前 6 位。
- 对这两个样本，固定 K=100 会多引入 94 个无关候选，另外还有 9 个 ambiguous ID。
- 一个待验证的假设：损失包含 EOS，答案最后一个位置只参与预测 EOS，监督信号可能偏弱。
- 这个假设还没有得到验证。

## 结论

- residual 的 token 筛选信号可以在多个样本上复现，不是单个样本的偶然结果。
- 不知道真实图像时，random surrogate 仍能保留 K=100 下的大部分收益。
- 真实图像对 token 排序帮助很小。
- 真实图像能改善问题部分的重建。
- 本页 rank 8 条件的瓶颈仍在第二阶段：固定宽度的累积 beam search 在 full coverage 时也排不出正确句子；后续 rank 32 在不修改搜索实现的情况下恢复部分完整问答（[[dager-lora-rank]]）。
- 答案部分的失败尤其明显。

## 局限

- 这是开发阶段的证据，不是独立验证。
- `dager_adapted` 的解码器不是论文原版，这里的结论不能直接套用到原版 DAGER。
- 被完整恢复的问题（Residual (GT) 4 个，Raw (GT) 2 个）可能来自重复的问题模板。
- 这些问题的恢复，不代表答案或整个样本被恢复。

## 下一步

- 诊断 beam search：真实前缀在每一层能否保留，以及在每一层的排名。
- 实现一个单独命名、忠于原论文的阈值解码器作为对照。
- token 层面可以继续检查：
    - public projection 残差的范数；
    - 梯度的奇异值谱；
    - 信号秩和噪声秩分别变化时的敏感性；
    - 候选集大小与 full coverage 的关系。
- 任何选择规则都要先在开发集上冻结，再做独立验证。
- 不能用参考 token 为每个样本单独挑选 K。

??? note "运行记录"
    - 所有任务都已完成。
    - 没有出现 OOM，也没有耗尽预算。
    - GPU 显存已释放。
