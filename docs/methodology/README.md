# Methodology

这里记录方法的原理、输入条件和方案迭代。先理解方法如何工作，再到 [Experiments](../experiments/README.md) 查看它在什么条件下有效。

当前研究重点是 **DAGER 改编版的分阶段文本恢复**。建议先读 [Residual v1](dager/residual-v1.md)：从梯度里的文本线索开始，用一个可手算的例子解释投影，再对应到实际筛选和序列搜索。

## 现有方法

| Method | Role | Mechanism | Status | Read |
|---|---|---|---|---|
| `dlg_adapted`、`ig_adapted` | 图像或联合重构 | 优化候选输入，使其产生的更新接近观测更新 | 已实现；适用性取决于具体条件 | [优化类方法](optimization.md) |
| TAG、LAMP 文本组件 | 优化式文本恢复 | 连续 token 表示，以及候选序列的离散调整 | 已实现；作为优化攻击中的文本组件使用 | [文本优化](optimization.md#text-components) |
| Raw | DAGER 打分对照 | 直接比较候选特征与上传梯度行空间 | 已实现，`mode=raw` | [Raw 对照](dager/raw.md) |
| Residual v1 | 当前文本恢复方案 | 同时投影梯度和候选，再筛选 token、搜索前缀 | 已实现，`mode=public_residual` | [原理与实现](dager/residual-v1.md) |
| 稀疏字典、视觉统计、逐 token 校准 | 第一阶段候选改进 | 尝试重新排序候选 token | 研究代码；未替代当前方案 | [改进方案](dager/refinements.md) |
| token–图像排序 | 机制诊断 | 检查候选 token 是否更能保持图像远近排序 | 使用真值的诊断；尚未成为攻击方法 | [研究边界](dager/refinements.md#image-ordering) |

其他已实现的优化变体及其与原论文的差异，见 [Baselines](../baselines.md)。实现与支持条件以 `attacks/registry.py`、文本组件和适用性检查为准；`dager` 原版入口仍为 `not_implemented`，不能与 `dager_adapted` 混称。

## 方案如何迭代

- **一个方法一个目录**：例如 `dager/` 保存这条方法路线的基线、当前方案和候选改进。
- **方法发生变化时新增版本页**：例如后续增加 `dager/residual-v2.md`，写清相对 v1 改了什么、为什么改、验证是否通过，并在目录页标出当前采用的版本。
- **解释性修订保留在当前页**：纠正术语、补充推导或改善图示，不意味着攻击算法升级。此处的 `v1` 是方法文档快照编号，不是新的代码发布版本。
- **实验结果链接到记录页**：开发集、独立验证和真值诊断分别标明。尚未验证的想法保持“候选”状态，不提前作为当前方案。
- **新页加入导航**：在 `mkdocs.yml` 的 Methodology 下登记，更新本页或对应方法目录的链接。写作结构见[方案迭代模板](iteration-template.md)。

## 阅读边界

- 攻击者只获得知识条件明确公开的字段；详见 [Protocol](../protocol.md)。
- 方法图和教学例子只解释机制，不代表实验成功率。
- Methodology 的版本表示算法方案变化；LoRA rank、候选预算和 beam 等实验条件变化，分别记录在配置和实验页中。
