---
id: tag-text-only-public-image
title: TAG 纯文本重建
date: 2026-09-29
track: text-tag
importance: supporting
data: slake/tune
model: llava-1.5-7b
fed: fedsgd
knowledge: [image_known]
token_lengths_known: true
n: 3
evidence_level: development
outcome: negative
uses_private_reference: false
contains_private_tokens: false
commits: [39b45e2, ec0f48e]
code: [run_yaml/slake_llava_image_known_lengths_states.yaml]
artifacts:
  - path: outputs/slake_llava_text_leakage/image_known_lengths_states/
related: [dager-single-sample-diagnostics]
---

## 要点

- **问题完全没有恢复**：18 次运行中，问题的 EM、ROUGE-1/L 和 word recall 全部为 0。
- **答案只有微弱信号**：只有 F-CL 第 0 轮完整恢复出 1 个答案（只有一个 token），另有 1 个部分恢复。
- **训练没有帮助**：到第 10 轮，F-C 和 F-L 仍然为 0，F-CL 的微弱信号也消失了。
- **损失下降不代表恢复**：TAG 目标函数持续下降，但重建文本和原文没有任何重叠。

## 研究问题

- **恢复结果**：已知真实图像和私有文本的长度时，TAG 能否恢复问题或答案？不同微调策略和训练轮次下，结果是否不同？
- **损失与恢复**：TAG 目标函数下降时，文本是否随之恢复？

## 实验设置

| Item | Setting |
|---|---|
| 攻击 | `tag_adapted`，1 个攻击种子，2000 次迭代，1 次 restart |
| 模型 | LLaVA-1.5-7B |
| 数据 | SLAKE `tune`，3 张图像 |
| 条件 | F-C、F-L、F-CL 三种策略 × 第 0、10 轮，共 6 个调度任务 |
| 联邦设置 | FedSGD |
| 攻击者知识 | `image_known`：知道真实图像<br>知道每个样本问题和答案的 token 长度<br>不知道问题和答案的内容 |
| 图像的使用 | 图像作为固定的 `Observation` 输入，不参与优化，也不参与打分 |

## 结果

### 恢复结果

**回答**：基本不能。问题在全部 18 次运行中都没有恢复；答案只有 F-CL 第 0 轮出现微弱信号，第 10 轮消失。

- 问题的 EM、ROUGE-1/L 和 word recall 在全部 18 次运行中都是 0。
- 答案指标只有 F-CL 第 0 轮不为 0：
    - 3 个答案中有 1 个完全恢复，这个答案只有一个 token。
    - 另有 1 个部分恢复：答案有 4 个 token，ROUGE-L 0.4，word recall 0.5。
- 随机起点下所有指标都是 0，但样本太少，无法区分真实泄露和短答案的偶然命中。
- 训练到第 10 轮没有帮助：F-C 和 F-L 仍为 0，F-CL 第 0 轮的信号也消失了。

| Strategy | Round | Question EM | Question ROUGE-L | Answer EM | Answer ROUGE-L | Answer word recall |
|---|---|---:|---:|---:|---:|---:|
| F-C | 0 | 0 | 0 | 0 | 0 | 0 |
| F-C | 10 | 0 | 0 | 0 | 0 | 0 |
| F-L | 0 | 0 | 0 | 0 | 0 | 0 |
| F-L | 10 | 0 | 0 | 0 | 0 | 0 |
| F-CL | 0 | 0 | 0 | 0.333 | 0.467 | 0.500 |
| F-CL | 10 | 0 | 0 | 0 | 0 | 0 |

### 损失与恢复

**回答**：没有。目标函数持续下降，但重建文本与原文没有任何重叠。

- 以 F-L 第 10 轮的 run 0 为例，原始 TAG 目标从 1482.93（第 50 次迭代）降到 1174.37（第 2000 次迭代）。
- 可观测的最佳离散得分出现在第 1250 次迭代（1.065，结束时为 1.413）。
- 但两个字段的重建结果与原文仍然完全没有重叠。
- 原始 TAG 目标没有归一化，不能跨条件比较。

## 结论

- 已知图像和长度时，TAG 基本无法恢复私有文本。
- 匹配损失下降并不意味着文本被恢复。

## 局限

- 这是开发阶段的证据，只有 3 张图像，不是 3 个独立的随机种子。
- 不能作为论文结果。

## 下一步

- 跑一组配对实验：`image_question_known` 加已知答案长度（问题公开，答案私有），覆盖 F-C、F-L、F-CL 的第 0、10 轮。
- 这组实验用来区分两种解释：是联合搜索失败了，还是文本目标本身没有信息量。
- 旧的调度 `run_yaml/slake_llava_image_question_known.yaml` 只跑过一个 20 次迭代的 pilot（`outputs/diagnostics/iqk_pilot`），尚未正式运行。

??? note "运行记录"
    - 6 个调度任务都正常结束（返回码 0）。
