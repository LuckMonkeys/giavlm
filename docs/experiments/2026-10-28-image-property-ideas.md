---
id: image-property-ideas
title: 图像属性方法验证
date: 2026-10-28
end_date: 2026-11-01
track: property
importance: supporting
data: slake
model: llava-1.5-7b
tuning: F-L
round: 10
fed: fedsgd
knowledge:
- private
- text_known
- image_known
n: 60
evidence_level: diagnostic
outcome: mixed
uses_private_reference: false
contains_private_tokens: false
uncommitted: true
commits:
- 2573d8e
- 2fd25b6
artifacts: []
related:
- gradient-prototype-transfer
- candidate-scoring
- visual-concept-probe
- convex-token-recovery
- synthetic-property
---

## 要点

- **深层梯度原型是唯一有正面信号的方向**：公开原型的第 32 层 Q 在 A、B、C 三批医学图像上都能判断头部；公开选定的第 24 层失败，见 [[gradient-prototype-transfer]]。
- **张量加权与类内抑制没有提高候选评分**：A 批二十图上，加权与等权基线同为 0.583，见 [[candidate-scoring]]。
- **首层词向量概念探针没有通过门槛**：主探针把 96 张公开图全判为 non-head；真实视觉 tokens 读出则有 61/72 和 24/24，见 [[visual-concept-probe]]。
- **凸组合 tokens 恢复不如直接选单个候选**：12 个公开查询上判对 9/12，最佳单候选的公开标签为 12/12，见 [[convex-token-recovery]]。
- **文本引导候选生成尚未执行**：还没有按恢复的问答生成场景相关的候选。

本页汇总[原始计划](../plans/2026-10-28-image-property-ideas.md)与[下一阶段计划](../plans/2026-10-28-image-property-next-stage.md)中五个 idea 的验证状态，细节都在各自的记录页。医学与通用候选库的基线见 [[synthetic-property]]。

## 关键数字

| Idea | Record | Main result | Status |
|---|---|---|---|
| Tensor Discriminability Weighting | [[candidate-scoring]] | A 批 0.583，等权 0.583 | 未优于基线 |
| Within-Class Variation Suppression | [[candidate-scoring]] | A 批 0.533，跨风格与等权相同 | 未优于基线 |
| Visual Subspace Concept Probing | [[visual-concept-probe]]、[[gradient-prototype-transfer]] | 首层失败；Q32 Head 0.750、0.700、0.854 | 头部有信号，待确认 |
| Visual Token Recovery | [[convex-token-recovery]] | 凸组合 9/12，单候选 12/12 | 工程通过，未超过单候选 |
| Text-Guided Candidate Generation | — | — | 未执行 |

## 研究问题

- **张量判别加权**：按张量的类间判别力加权梯度距离，能否改善部位判断？
- **类内变化抑制**：去掉候选库的类内变化方向，能否改善部位判断和跨风格迁移？
- **视觉子空间概念探针**：能否用视觉特征或梯度空间里的概念原型直接判断部位？
- **视觉 tokens 恢复**：能否通过梯度匹配恢复视觉 tokens，再读出部位？
- **文本引导候选生成**：能否按恢复的问答生成场景相关的候选，以改善判断？

## 实验设置

| Item | Setting |
|---|---|
| Model | LLaVA-1.5-7B，F-L，LoRA rank 32，round 10，单样本单步 FedSGD |
| Medical cohorts | SLAKE 的 A、B、C 三批，各 20 张；定义见 [[gradient-prototype-transfer]] |
| Public data | 摄影、线稿、黏土渲染三种风格的非医学合成图，以及医学外观的合成候选 |
| Selection boundary | 各页的方法、层与参数都只用公开数据选择；医学标签在全部预测提交后才读取 |
| Records | 每个 idea 的设置、结果与运行记录都在对应页面；本页不重复 |

## 结果

### 张量判别加权

**回答**：不能。公开选出的加权（`weighted_b0`）与等权基线同为 0.583；打乱标签或置换权重后结果也不变。

- 跨风格公开开发中，加权在三种留出风格上与等权完全相同，未通过门槛。
- 详见 [[candidate-scoring]]。

### 类内变化抑制

**回答**：不能。类内抑制与 Fisher 都为 0.533，低于等权的 0.583；抑制相对未抑制原型聚合的差异区间跨零。

- 抑制在去掉类内变化的同时也去掉了类间间隔；最强配置只保留 42.5%。
- 详见 [[candidate-scoring]]。

### 视觉子空间概念探针

**回答**：首层词向量探针失败；改用深层梯度原型后，第 32 层 Q 能在三批医学图像上判断头部，但公开选定的第 24 层和腹部判断都失败。

- 首层探针的问题更可能在词向量构造：真实视觉 tokens 读出时，摄影组 61/72、医学组 24/24 判对，见 [[visual-concept-probe]]。
- Q32 在 A 批 CT、MRI 内部分别判对 6/8 和 8/8，不能用模态解释；多层聚合、多问答校准和换用 K、V 都没有帮助，见 [[gradient-prototype-transfer]]。

### 视觉 tokens 恢复

**回答**：工程上可行，但在受控公开任务上不如直接选单个候选：凸组合判对 9/12，最佳单候选 11/12，其公开标签 12/12。

- 二阶梯度链与注入接口的误差为零；优化比不看上传的字典均值高 0.250。
- 详见 [[convex-token-recovery]]。

### 文本引导候选生成

**回答**：尚未执行，没有结果。

- 计划先按恢复文本中的场景线索分组，再生成条件候选，见[原始计划](../plans/2026-10-28-image-property-ideas.md)。

## 结论

- 五个 idea 中，只有深层梯度原型在真实医学图像上留下了可重复的头部信号（Q32）。
- 候选评分的统计改进和凸组合恢复都没有超过简单基线。
- 首层词向量探针和公开选定的主条件失败；可能的原因分别是探针构造，以及公开图与医学图捕获时的问答差异，后者尚未单独检验。

## 局限

- 三批医学图像各只有 20 张，患者独立性未知，部位与模态部分相关。
- 公开开发数据都是合成图，且多用固定已知的问答。
- 各方法的完整限制见对应记录页。

## 下一步

- 优先把 Q32 头部判断作为预先声明的主条件，在新的确认批次上验证，见 [[gradient-prototype-transfer]]。
- 文本引导候选生成可在上述确认之后再决定是否启动。

??? note "运行记录"

    - 2026-11-02 将本页按研究问题拆为四页：[[gradient-prototype-transfer]]、[[candidate-scoring]]、[[visual-concept-probe]]、[[convex-token-recovery]]。各阶段的 GPU、PID、失败尝试与测试计数随内容迁到对应页面的运行记录。
    - 拆分前的完整版本未提交 Git，备份在 `outputs/diagnostics/docs_backup_20261102/2026-10-28-image-property-ideas.md`。
