# DAGER 文本恢复：方案与迭代

当前实现名为 **`dager_adapted`**。它面向单样本 VLM 梯度，先从词表中选出可疑 token，再利用候选前缀的上下文特征搜索顺序。这里记录的是项目的 LoRA 与 VLM 改编方案。

## 建议阅读顺序

1. [Raw 对照](raw.md)：为什么上传梯度能提供文本线索，直接用梯度空间打分有什么问题。
2. [Residual v1](residual-v1.md)：如何同时处理梯度和候选，以及为什么纯文本向量也要做投影。
3. [候选改进与验证状态](refinements.md)：稀疏字典、视觉统计、校准和图像排序分别尝试改变哪一步。

## 当前方案位置

| Scheme | Change | Status | Evidence |
|---|---|---|---|
| Raw | 使用原梯度行空间 | 保留作对照 | [[dager-multisample-n10]] |
| [Residual v1](residual-v1.md) | 从梯度与候选中同时去掉已知输入方向 | 当前采用的打分机制 | [[dager-multisample-n10]]、[[dager-lora-rank]] |
| [第一阶段候选改进](refinements.md) | 修改 token 排序或候选证据 | 研究变体，未替代 Residual | [[dager-filter-phase]] |
| [纯文本语言先验](text-prior.md) | 在第二阶段加入内容 NLL 软惩罚 | 本轮冻结验证有提升，默认关闭 | [[dager-lm-prior]] |
| [图像排序信号](refinements.md#image-ordering) | 利用 token 与图像排序之间的关系 | 真值诊断；待独立验证 | [[token-image-ordering]] |

`Residual v1` 是对现有实现的第一份方法快照，记录日期为 2026-10-11。它没有改变攻击默认行为。后续如果修改算法，再新增版本页并记录与本页的关系；提高 LoRA rank 或 beam 预算本身不算 Residual 的新版本。

## 所有版本都要区分的两个结果

- **候选覆盖**：真实问答中的 token ID 是否都在候选集合里。这个集合不提供位置、顺序或重复次数。
- **序列恢复**：最终问题和答案是否与真值一致。即使候选全覆盖，打分误差和 beam 剪枝仍会让正确序列丢失。

结果评估在重构保存之后读取真值。代码中的 `completed` 表示得到并保存了完整长度的候选，不表示攻击已经恢复真值。
