# 纯文本语言先验：辅助序列搜索

## 要点

- **本轮改变**：在第二阶段累计 Residual 距离上增加语言模型的内容 token NLL。
- **采用状态**：本轮冻结验证有提升；`lm_prior_weight=0` 仍是默认值，实现已按用户要求提交，见 `9efa65d`。
- **与上一版的关系**：沿用 [Residual v1](residual-v1.md) 的投影、候选筛选和几何评分，不改变第一阶段候选集合。

## 研究问题

真实 token 已经包含在候选集中时，纯几何评分仍可能保留错误序列。语言先验尝试为顺序提供额外偏好。它不能证明某个词属于私有文本，也不负责预测真实图像中的事实。

## 输入条件

| Item | Setting |
|---|---|
| Implementation | `dager_adapted`；`attack.dager.lm_prior_weight` |
| Observation | 与 Residual v1 相同 |
| Prior model | 公开受害模型内的语言模块，权重冻结 |
| Prior inputs | 公开文本模板及当前候选文本；公开问题条件下可使用声明公开的问题 |
| Visual context | Residual 路径沿用原图像；先验路径完全不接收图像 |
| Private reference | 攻击不读取；实验评估仅在重构保存后读取 |

## 原理与流程

设候选序列前缀为 `w₁…wₜ`，`dᵢ` 是新增第 i 个 token 时的原 Residual 距离。新评分为：

```text
S(w₁…wₜ) = Σᵢ dᵢ + λ Σᵢ [−log p_text(wᵢ | w₁…wᵢ₋₁, public template)]
```

- 分数越小越优；λ 是非负先验权重，λ=0 精确退化为原始搜索。
- 先验采用全词表 log-softmax，不把候选集合重新归一化。
- 语言模型读取父前缀，预测尚未追加的 token；不能拿追加 token 后的 logits 给该 token 打分。
- 内容 token 参与累计 NLL；固定模板只提供上下文。先验路径不使用视觉槽位、EOS 或 padding。
- 同一父前缀只做一次完整纯文本前向，得到全部候选的概率，然后与 Residual 距离组合并保留 beam。
- 不按 Top-M 或概率阈值预先删除 token；低概率增加有限代价，仍可能被梯度证据保留。

这是“观测匹配与语言先验”形式的正则化。Residual 分数不是已经校准的负对数似然，因此不声称精确贝叶斯后验或必然恢复。

语言先验用于文本梯度反演的思路可参考 [LAMP](https://arxiv.org/abs/2202.08827)。本变体使用受害模型自身的纯文本路径，不是对完整 LAMP 算法的复现。

## 与上一版的区别

| Component | Residual v1 | Text prior | Motivation |
|---|---|---|---|
| Token filter | 投影后距离 | 保持相同 | 单独评估序列阶段的变化 |
| Prefix score | 累计距离 | 累计距离加内容 NLL 惩罚 | 引入文本顺序偏好 |
| Extra forward | 无 | 每个父前缀一次完整纯文本前向 | 取得 next-token 概率 |
| Backward | 可选梯度重排 | 本实验不使用 | 避免同时更改两个评分环节 |
| Checkpoint | schema 1 | schema 2 | 保存当前深度的先验分数及分项成本 |

## 验证与结论

实验设置、保存状态诊断和最终结果集中在 [[dager-lm-prior]]。rank 32、未知图像、公开 token 长度条件下，冻结权重 0.01 在十个新样本上提高了精确恢复率。默认仍为 0，不能将本轮小样本结果解释成恢复保证。

相同权重在 rank 8 的原十样本上仅将 Joint EM 从 0/10 提高到 2/10，两个成功样本对应同一种问答。四个样本候选不完整，另有四个候选完整的样本仍发生问题前缀剪枝。因此不能把 rank 32 的收益视为先验跨 rank 的普遍效果，也不能据此断言只要加大权重就能恢复。

## 局限与下一步

- 医学缩写、生僻词和真实但不常见的答案可能被语言先验压低。
- 完整语言模型前向有额外成本；原搜索只运行第一个 Transformer block，不能声称 logits 免费可得。
- 软评分仍可能通过有限 beam 剪掉真实前缀；需要同时报告 EM、前缀保留率和耗时。

## 实现与修订记录

| Date | Revision | Description | Evidence |
|---|---|---|---|
| 2026-10-11 | 实验变体 | 默认关闭的纯文本 NLL 软评分；GPU 5 配对验证 | [[dager-lm-prior]] |
| 2026-10-12 | 冻结验证 | 新问答验证完成；保留最终排序失败和条件限制，默认值不变 | [[dager-lm-prior]] |
