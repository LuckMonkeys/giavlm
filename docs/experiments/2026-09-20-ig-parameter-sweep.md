---
id: ig-parameter-sweep
title: IG 超参数扫描
date: 2026-09-20
track: image-ig
importance: archive
data: slake/tune
model: llava-1.5-7b
tuning: F-C
round: 0
fed: fedsgd
knowledge: [text_known]
n: 3
evidence_level: diagnostic
outcome: negative
uses_private_reference: false
contains_private_tokens: false
archive_note: >-
  这批实验的候选图像是 bf16，小步长的更新被舍入掉了；引用前需要用 float32 候选图像重跑。
commits: [1e1d328]
code: [run_yaml]
artifacts:
  - path: outputs/slake_llava_ig_parameter_sweep/lr_report.json
  - path: outputs/slake_llava_ig_parameter_sweep/tv_report.json
  - path: outputs/slake_llava_ig_parameter_sweep/iterations_report.json
  - path: outputs/slake_llava_ig_parameter_sweep/sweep_report.json
  - path: docs/ppt/ig_parameter_sweep_summary.xlsx
    note: 仅在本地（docs/ppt 不进 git）
related: [ig-failure-diagnosis]
---

## 要点

- **所有设置都停留在噪声水平**：12 种设置下 PSNR 在 6.5–7.8 dB 之间，SSIM 约为 0.005–0.007。
- **调参不是瓶颈**：改变学习率、TV 权重或迭代次数，重建都没有明显变好。
- **结果需要重跑才能引用**：候选图像是 bf16，学习率 0.001 的更新大多被舍入掉了。

## 研究问题

- **超参数**：IG 在 SLAKE / LLaVA 上失败，是不是因为攻击超参数没调好？

## 实验设置

| Item | Setting |
|---|---|
| 模型 | LLaVA-1.5-7B |
| 数据 | SLAKE `tune`，3 张图像 |
| 联邦设置 | FedSGD，batch 1，第 0 轮 |
| 上传梯度 | F-C（connector） |
| 攻击 | `ig_adapted`，1 次 restart<br>`attack.text_method=none` |
| 攻击者知识 | `text_known`：知道问题和答案 |
| 扫描方式 | 每次只改一个参数，其余保持默认<br>12 种设置 × 3 张图像，共 36 次运行 |
| 指标 | MSE、PSNR、SSIM（未启用 LPIPS 和 CLIP） |

## 结果

- 学习率越大，PSNR 越低，但所有取值都在噪声水平。
- TV 权重和迭代次数几乎不影响 PSNR。
- SSIM 在所有设置下都接近 0：LR 扫描约 0.005–0.007，TV 扫描约 0.007，迭代次数扫描为 0.0071–0.0074。

| Parameter | Value | PSNR |
|---|---|---:|
| Attack LR | 0.001 | 7.74 |
| Attack LR | 0.003 | 7.73 |
| Attack LR | 0.01 | 7.57 |
| Attack LR | 0.03 | 7.09 |
| Attack LR | 0.1 | 6.52 |
| TV weight | 0 | 7.73 |
| TV weight | 0.001 | 7.73 |
| TV weight | 0.01 | 7.73 |
| TV weight | 0.1 | 7.78 |
| Iterations | 500 | 7.77 |
| Iterations | 1000 | 7.78 |
| Iterations | 2000 | 7.80 |

## 结论

- 所有设置都处于噪声水平，调参不是 IG 失败的原因。
- 后续实验采用的工作点：LR 0.003，TV 0.1，2000 次迭代。

## 局限

- 当时还没有 `attack.image_dtype`，候选图像是 bf16，LR 0.001 的更新大多被舍入掉了。
- 引用前需要用 float32 候选图像重跑；设置 `attack.image_dtype=bfloat16` 可以复现旧结果。
- 只有 3 张图像，只能作为诊断，不足以支撑论文结论。

## 下一步

- 用 float32 候选图像重跑随机起点的 IG 基线。

??? note "运行记录"
    - 这批捕获是 Observation v4，当前的 `attack` 和 `diagnose-gradient` 会拒绝读取；要复用需要重新捕获（确定性，几分钟）。
    - 100,000 次迭代的后续实验（`run_yaml/slake_llava_ig_iteration_occ.yaml`）在第 15150 次迭代时中断，调度器里显示 `running` 的那一行已经过期。
