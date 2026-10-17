---
id: qwen-image-generation
title: Qwen-Image 本地生成测试：医学候选图像的外观与模态可信度
date: 2026-10-16
track: discriminability
importance: supporting
data: synthetic-prompts
model: Qwen-Image-2.1-Turbo
knowledge: [private]
n: 10
evidence_level: smoke
outcome: mixed
uses_private_reference: false
contains_private_tokens: false
uncommitted: false
commits: [92169af]
code: [evaluation/qwen_image_generation_smoke.py]
artifacts:
  - path: docs/experiments/data/qwen-image-generation/aggregate.json
    note: 纳入 Git 的聚合指标与本地来源哈希，不含逐样本文本、图像或原始标识
  - path: outputs/diagnostics/qwen_image_gpu5_20261016/plan.json
    note: 生成前保存的提示词、种子、参数、软件版本和代码哈希
  - path: outputs/diagnostics/qwen_image_gpu5_20261016/complete.json
    note: 十张生成图的哈希、尺寸与逐图耗时
  - path: outputs/diagnostics/qwen_image_gpu5_20261016/integrity_audit.json
    note: 图像完整性、提示词一致性、代码哈希与时间汇总
  - path: outputs/diagnostics/qwen_image_gpu5_20261016/visual_review.json
    note: 目视筛查记录，不是临床验证或模态分类准确率
related: [image-property-inference, gradient-discriminability]
---

## 要点

- **本地文生图模型已跑通**：十张 1024 × 1024 图像均生成成功，平均每张生成与保存耗时约 3.98 秒。
- **普通场景的中英文描述均得到遵循**：两张控制图都呈现左侧红色立方体、右侧蓝色球体，数量与颜色符合提示词。
- **医学外观可生成，但 MRI 标签尚不可信**：CT 和 X-Ray 呈现预期的大类外观；两张腹部 MRI 输出仍偏 CT 风格，不能直接作为可靠 MRI 原型。
- **本轮没有测试梯度区分性**：仅检查候选生成能力，没有读取私有样本、回放受害模型或计算梯度匹配分数。

## 关键数字

| Metric | Value |
|---|---:|
| Generated images | 10/10 |
| Resolution | 1024 × 1024 |
| Model initialization seconds | 15.99 |
| Total generation and save seconds | 39.79 |
| Mean generation and save seconds | 3.98 |
| Min generation and save seconds | 3.73 |
| Max generation and save seconds | 4.41 |
| Total script seconds | 76.72 |

总脚本时间包括依赖导入、初始化、生成、拼图和关闭模型。相同提示词的第二个种子可能复用文本编码，以上不是每张图都重新加载模型的耗时。

## 研究问题

用户提供的本地 Qwen 文生图模型是否能够按描述生成图像，特别是能否为后续模态推断提供 CT、MRI、X-Ray 合成候选？先验证生成质量，再决定是否接入已有梯度评分。

## 实验设置

| Item | Setting |
|---|---|
| Model | 本地 Qwen-Image-2.1-Turbo |
| Runtime | 已安装的 qwen-image-21 环境；vLLM-Omni |
| GPU | 用户指定的 GPU 5，A100-SXM4-80GB |
| Precision | bfloat16 |
| Sampling | 原生 8 步 sample_sigmas；CFG 1.0 |
| Resolution | 1024 × 1024 |
| Execution | 单图顺序生成，模型加载一次，enforce_eager=true |
| Control prompts | 同一日常场景，英文与中文各一张，seed 42 |
| Medical prompts | 腹部 CT、腹部 T2 MRI、胸部 CT、胸部 X-Ray，各 seed 42、43 |
| Private inputs | 无；全部提示词独立编写，没有查看或描述目标私有图像 |
| Victim gradients | 未使用 |

前置元数据的 `knowledge: private` 表示没有提供私有字段；本轮属于候选生成 smoke test，没有构建攻击 Observation，也不是 private 条件下的攻击效果评测。`n=10` 指生成图数，不是十个受害者样本。

腹部 CT 与 MRI 提示词尽量保持部位、切面和构图一致，仅调整成像描述；相同种子仍不保证两种提示生成相同解剖结构。胸部 CT 为横断面，X-Ray 为正面投影，因此它们不能构成仅改变模态的严格配对实验。

## 结果

### 描述遵循与医学外观

- 普通场景的物体数量、颜色、左右关系均符合要求。
- 医学输出均呈灰阶外观，未见明显文字、水印、箭头或多图拼接。
- 腹部 MRI 两张原图的中央骨样结构亮白，整体与 CT 输出相似，未通过本轮目视可信度筛查；这不是专业模态鉴定。
- 医学图像的器官形状、空间关系、具体 MRI 序列或 X-Ray 投照几何均未获得独立验证。

| Prompt | Images | Mean seconds | Visual assessment |
|---|---:|---:|---|
| Control English | 1 | 4.41 | 数量、颜色、位置符合 |
| Control Chinese | 1 | 4.05 | 数量、颜色、位置符合 |
| Abdomen CT | 2 | 3.88 | 腹部横断面，呈 CT 样式；解剖真实性未验证 |
| Abdomen MRI | 2 | 3.74 | 腹部横断面，但偏 CT 样式；MRI 标签待核验 |
| Chest CT | 2 | 3.93 | 肺野横断面，呈 CT 样式 |
| Chest X-Ray | 2 | 4.12 | 胸部正面投影，呈 X-Ray 样式 |

所有原图、逐图提示词、种子和 JSON 记录保留在 `outputs/diagnostics/qwen_image_gpu5_20261016/`。`contact_sheet.jpg` 是十图预览，`medical_preview.jpg` 是四类医学提示的首个种子预览；均未复制到文档站点。

### 完整性检查

- 十张 PNG 均可解码，尺寸符合计划，保存后的 SHA-256 与逐图记录一致。
- 十张图像的哈希互不相同；实际提示词与种子集合和生成前计划一致。
- 生成代码哈希与运行计划一致，未发生失败重试或参数降级。
- 所有数字与目视结论来自上方列出的本地产物；评估代码已提交，原始产物保存在本地。

## 结论

- 可以用该模型快速构造合成候选图像，普通提示的遵循效果在本次小样本上较好。
- 不能把提示词中的属性直接视为生成图像的真实标签；MRI 输出是当前最明确的风险点。
- 现有结果尚不能证明合成候选可以提高梯度模态推断准确率，也不能替代真实公开原型库结果。

## 局限

- 每种医学提示只有两个种子，没有系统评估多样性、模态准确率或合成与真实图像的分布差异。
- 目视筛查由助手完成，没有医学专家或独立模态标注复核。
- 相同提示和种子之间的外观差异不代表具有足够的病例多样性。
- 没有评估生成器训练集与目标数据的潜在重叠；仅能确认本轮生成输入未使用私有参考。

## 下一步

- 先完善并独立核验同部位 CT 与 MRI 的生成提示；不要将本批两张 MRI 直接加入可靠标签库。
- 冻结通过核验的候选库与类别标签，不依据目标梯度分数挑选看起来更成功的生成样本。
- 在相同目标观测、相同重构文本和相同评分规则下，比较公开真实原型库与合成原型库；保留 Model-only、校准 Text-only 和错配梯度对照。
- 先使用正确文本诊断候选库是否具备信号，再使用已保存的实际重构文本。真实目标数量不足的问题仍需通过真实独立数据解决。

??? note "运行记录"

    - 日期：2026-10-16；GPU 5；worker PID 1242426、watcher PID 1242427 均已退出，显存回到启动前约 20 GB 的既有占用。
    - 复用用户已配置的专用环境与本地权重；离线运行，缓存写入 `/tmp`，未修改模型目录、核心攻击代码或默认参数。
    - 启动脚本：`outputs/diagnostics/analysis_scripts/launch_qwen_image_gpu5_20261016.sh`；日志：`outputs/diagnostics/qwen_image_gpu5_20261016.log`。
    - 原生采样步数由模型配置加载；没有额外设置 num_inference_steps，没有使用负面提示。
    - 本轮无失败、无 OOM、无重试。新增生成脚本 Ruff 通过；十图产物检查通过。
