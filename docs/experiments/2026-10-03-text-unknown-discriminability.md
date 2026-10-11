---
id: text-unknown-discriminability
title: 文本未知时的梯度-图像可区分性
date: 2026-10-03
track: discriminability
importance: key
data: slake/tune
model: llava-1.5-7b
fed: fedsgd
knowledge: [text_known, question_known, target_known, private]
n: 20
evidence_level: development
outcome: negative
uses_private_reference: true
private_reference_note: 候选集包含真实图像，排序好坏按与真实图像的误差判断
contains_private_tokens: false
commits: [ca4e96c, eb2a799, 355cefa, d214976, b688359, 73ebac2]
code:
  - evaluation/text_unknown_discriminability.py
  - configs/diagnostics/slake_llava_text_unknown_discriminability.yaml
artifacts:
  - path: outputs/diagnostics/text_unknown_discriminability/
    note: 研究根目录；development-status.log、development-watch.log
  - path: outputs/diagnostics/text_unknown_discriminability/reports/pilot/f_l_r10/
  - path: outputs/diagnostics/text_unknown_discriminability/reports/development/f_c_r0/
  - path: outputs/diagnostics/text_unknown_discriminability/reports/development/f_l_r10/
  - path: outputs/diagnostics/text_unknown_discriminability/reports/development/f_cl_r10/
  - path: outputs/diagnostics/text_unknown_discriminability/reports/development/comparison/
    note: loss_vs_mse_raw 和 loss_rank_vs_mse（主要对比用）两张图
  - path: outputs/diagnostics/analysis_scripts/plot_text_unknown_loss_mse.py
  - path: outputs/diagnostics/text_unknown_discriminability_failed_20261003_091114/
    note: 失败的第一次 pilot（YAML 把 yes/no 解析成了布尔值）
related: [gradient-discriminability]
---

## 要点

- **文本已知时排序很强**：三个条件下 Spearman rho 为 0.66–0.85，near-win 为 0.86–0.97。
- **只知道答案时还保留部分排序**：训练后的 F-L、F-CL，rho 分别为 0.363 和 0.248。
- **只知道问题时排序反转**：F-L、F-CL 的 rho 分别为 −0.248 和 −0.220。
- **问题和答案都靠猜时排序消失**：F-L 的 rho 为 −0.005，接近随机；F-CL 为 −0.118，呈弱负相关。
- **未训练的 F-C 保留了弱信号，但主要依赖模板**：只用随机词猜测时，rho 只有 0.082，near-win 为 0.540。

## 关键数字

开发集（20 张图像），各条件下的 Spearman rho（损失排名与图像误差排名的相关性）：

| Strategy | Round | text_known | question_known | target_known | private |
|---|---|---:|---:|---:|---:|
| F-C | 0 | 0.658 | 0.529 | 0.569 | 0.299 |
| F-L | 10 | 0.846 | −0.248 | 0.363 | −0.005 |
| F-CL | 10 | 0.800 | −0.220 | 0.248 | −0.118 |

## 研究问题

- 攻击者需要猜问题、答案或两者时，匹配损失还能否给候选图像正确排序？

## 实验设置

| Item | Setting |
|---|---|
| 性质 | 使用私有数据的诊断，不是攻击基准 |
| 命令 | `python -m core.commands diagnose-text-unknown-discriminability` |
| 模型 | LLaVA-1.5-7B |
| 条件 | F-C 第 0 轮<br>F-L 第 10 轮<br>F-CL 第 10 轮 |
| 攻击者知识 | `text_known`：问题和答案都知道<br>`question_known`：只知道问题<br>`target_known`：只知道答案<br>`private`：问题和答案都不知道 |
| 候选集 | 25 张图像；每个文本猜测在整个候选集上保持不变 |
| 文本猜测 | 4 个公开模板，加上 4 个确定性的随机词猜测 |
| 聚合方式 | 对各个猜测下的排名取平均（mean-rank aggregate） |
| 样本 | pilot：F-L 第 10 轮，3 张图像<br>开发集：20 张图像 |
| 隐私保护 | 报告中只保留不透明的猜测 ID，不包含私有文本，也不包含猜测的文本 |

## 结果

### 开发集（20 张图像）

- 知道问题和答案时，三个条件都有很强的排序。
- 训练后的 F-L 和 F-CL 强烈依赖正确的答案：只知道答案时排序为正，只知道问题时排序反转。
- 问题和答案都靠猜时，F-L 接近随机，F-CL 呈弱负相关。
- 未训练的 F-C 在各种知识条件下都保留了一定排序，但这主要来自模板。

Near-win（图像误差更小的候选，损失也更低的概率；0.5 为随机）：

| Strategy | Round | text_known | question_known | target_known | private |
|---|---|---:|---:|---:|---:|
| F-C | 0 | 0.864 | 0.793 | 0.811 | 0.692 |
| F-L | 10 | 0.966 | 0.348 | 0.765 | 0.547 |
| F-CL | 10 | 0.950 | 0.375 | 0.649 | 0.391 |

下图把每个条件下损失排名与图像误差的关系画了出来，是本研究的主要对比图：

- 每行是一个模型状态，每列是一种攻击者知识；点是候选图像，红线和色带是分箱中位数和四分位范围。
- 损失先在每个固定的文本猜测内排名，再在各个猜测间取平均，以消除不同猜测带来的损失尺度差异。
- 曲线越往右上升，说明损失越能把误差小的图像排在前面；平线表示没有排序能力。

[![Loss rank vs image MSE](figures/text-unknown-discriminability/loss_rank_vs_mse.png)](figures/text-unknown-discriminability/loss_rank_vs_mse.png)

??? note "原始损失与图像误差（未控制尺度，仅作参考）"
    - 纵轴是原始余弦损失；不同文本猜测下损失的尺度不同，所以不能直接跨列比较。
    - 点的颜色区分正确文本、模板猜测和随机词猜测；黑线和灰色带是分箱中位数和四分位范围。

    [![Raw loss vs image MSE](figures/text-unknown-discriminability/loss_vs_mse_raw.png)](figures/text-unknown-discriminability/loss_vs_mse_raw.png)

### Pilot（F-L 第 10 轮，3 张图像）

- pilot 的结果方向与开发集一致。
- Truth-image percentile 是真实图像在候选集中的排名百分位，越低越好。

| Knowledge | Spearman rho | Near-win | Truth-image percentile |
|---|---:|---:|---:|
| text_known | 0.865 | 0.980 | 0.000 |
| question_known | −0.339 | 0.263 | 0.568 |
| target_known | 0.252 | 0.714 | 0.497 |
| private | 0.025 | 0.528 | 0.498 |

## 结论

- 知道正确的问题和答案时，所有条件下都有很强的图像排序。
- 训练后的模型，图像排序强烈依赖正确的答案。
- 文本完全未知时，匹配损失不能提供稳定、与模型无关的图像排序信号。

## 局限

- 这是开发阶段的证据，独立验证还没有开始。
- 比较时应以控制尺度的 `loss_rank_vs_mse` 图为主，因为不同猜测文本下原始损失的尺度不同。

## 下一步

- 冻结开发阶段的分析方法，再在独立样本上验证。

??? note "运行记录"
    - pilot：1,875/1,875 条记录，用时 997 秒；GPU 峰值显存 allocated 20.14 GiB，reserved 20.83 GiB。
    - 开发集：物理 GPU 4，37,500/37,500 条记录（每个条件 12,500 条）；所有报告都已完成，没有失败记录。
    - 第一次 pilot 在第 425 条记录时中断，因为 YAML 把未加引号的 `yes`、`no` 词表项解析成了布尔值。commit `eb2a799` 给它们加了引号，并让读取配置时拒绝非字符串的模板和词表。
