---
id: dager-filter-phase
title: DAGER token 筛选改进
date: 2026-10-05
track: text-dager
importance: key
data: slake/tune+eval
model: llava-1.5-7b
tuning: F-L
round: 10
fed: fedsgd
knowledge: [image_known, private]
token_lengths_known: true
n: 50
evidence_level: validation
outcome: negative
uses_private_reference: true
private_reference_note: 仅稀疏字典的“数值诊断”一节用私有的前向和反向量重算梯度
contains_private_tokens: true
commits: [7be400e]
code:
  - attacks/analytic/dager_filter_refinement.py
  - evaluation/dager_filter_study.py
  - evaluation/dager_sparse_filter_study.py
  - evaluation/dager_filter_validation.py
  - evaluation/dager_filter_position_diagnostic.py
  - attacks/analytic/dager_visual_filter.py
  - evaluation/dager_visual_filter_study.py
  - evaluation/dager_visual_filter_followup.py
  - attacks/analytic/dager_evidence.py
  - evaluation/dager_evidence_study.py
  - evaluation/dager_evidence_validation.py
artifacts:
  - path: outputs/diagnostics/dager_filter_findings_20261005.json
    note: 稀疏字典：汇总结论
  - path: outputs/diagnostics/dager_filter_gpu5_20261005_v2/
    note: 稀疏字典：开发集消融（40 个条件）
  - path: outputs/diagnostics/dager_filter_gpu5_20261005/
    note: 稀疏字典：失败的第一次尝试（CPU/GPU 张量放置），已保留
  - path: outputs/diagnostics/dager_filter_followup_gpu5_20261005/
    note: 稀疏字典：使用私有数据的位置诊断，以及更长的稀疏求解
  - path: outputs/diagnostics/dager_filter_validation_gpu5_20261005/
    note: 稀疏字典：冻结验证（plan.json, report.json, complete.json）
  - path: outputs/diagnostics/dager_visual_filter_findings_20261005.json
    note: 公开图像统计加权：汇总结论
  - path: outputs/diagnostics/dager_visual_filter_gpu5_20261005/
    note: 公开图像统计加权：第一轮开发
  - path: outputs/diagnostics/dager_visual_filter_extended_gpu5_20261005/
    note: 公开图像统计加权：扩展 ridge 范围
  - path: outputs/diagnostics/dager_visual_filter_validation_gpu5_20261005/
    note: 公开图像统计加权：冻结验证
  - path: outputs/diagnostics/dager_evidence_gpu5_20261005_v5/
    note: 逐 token 校准：开发集（v5；base、v2、v3、v4 的失败尝试已保留）
  - path: outputs/diagnostics/dager_evidence_validation_gpu5_20261005/
    note: 逐 token 校准：冻结验证，offset 20–39
  - path: outputs/diagnostics/dager_evidence_replication_gpu5_20261005/
    note: 逐 token 校准：随机复现，种子 7391，offset > 40
  - path: outputs/diagnostics/dager_evidence_findings_20261005.json
  - path: outputs/diagnostics/dager_evidence_case_audit_20261005.json
  - path: outputs/diagnostics/dager_evidence_replication_case_audit_20261005.json
related: [dager-multisample, dager-single-sample-diagnostics]
---

## 要点

- **没有一种方法能稳定替代现有筛选器**：三种方法在开发集上都有提升，但在新样本上要么只在特定条件下有效，要么退化。
- **稀疏字典排序只在已知图像时有效**：已知真实图像时，验证集 full coverage 从 5/10 升到 9/10；改用 random surrogate 后从 5/10 降到 2/10。
- **公开图像统计加权没有通过验证**：开发集 full coverage 从 6/10 升到 8/10，但验证集从 8/10 降到 6/10。
- **逐 token 校准只有局部提升**：40 张验证图像上 full coverage 从 33/40 升到 34/40，但随机复现时没有提升，recall 还略有下降。
- **决定保留现有筛选器**：不声称句子恢复有任何改进，下一步转向诊断解码器。

## 关键数字

Informative K=50（实际 59 个候选，含 9 个 ambiguous ID）；箭头表示基线 → 新方法：

| Method | Image | Dev full coverage | Validation full coverage | Validation recall | Verdict |
|---|---|---|---|---|---|
| Sparse dictionary | GT | 6/10 → 10/10 | 5/10 → 9/10 | 0.958 → 0.992 | 只在已知图像时有效 |
| Sparse dictionary | Random | – | 5/10 → 2/10 | 0.933 → 0.695 | 退化 |
| Random-surrogate joint fit | Random | 6/10 → 8/10 | 8/10 → 5/10 | 0.988 → 0.924 | 退化 |
| Public-visual joint fit | Random | 6/10 → 8/10 | 8/10 → 6/10 | 0.988 → 0.931 | 退化 |
| Null-rank mixture | Random | 6/10 → 6/10 | 33/40 → 34/40 | 0.981 → 0.985 | 复现时没有提升 |

## 研究问题

- **稀疏字典排序**：用 group lasso 让少数候选共同解释梯度，覆盖能否提高并在新样本上保持？
- **公开图像统计加权**：图像私有时，用公开图像的视觉统计代替随机替代图，覆盖能否提高并在新样本上保持？
- **逐 token 校准**：按每个 token 的 null 分布校准距离，覆盖能否提高并在新样本上保持？

三种方法都先在开发集上冻结，再在新的独立样本上验证；目标是在可控的候选数量下完整覆盖私有 token 集合。

## 实验设置

| Item | Setting |
|---|---|
| 背景 | 用户 2026-10-05 决定：先改进第一阶段 token 筛选，再改进句子搜索 |
| 授权 | 经用户授权在 GPU 5 上运行 |
| 受害模型 | LLaVA-1.5-7B，F-L 第 10 轮快照，LoRA rank 8，bf16<br>筛选分析用 float64 计算 |
| 开发集 | 原来的 10 个配对样本（SLAKE `tune`，offset 0–9） |
| 验证流程 | 每种方法先在开发集上调好参数并冻结，再在新的 `eval` 图像组上验证 |
| 现有筛选器（baseline） | 对词表中的每个 token，取它在第一层的输入向量，先去掉公开信息（模板文本、图像）张成的方向，再计算它到梯度行空间（Q、K、V 三组 LoRA-A 梯度合在一起）的距离；距离越小，越可能出现在私有文本中<br>依据：第一层权重梯度的行空间由实际输入 token 的向量张成；每个 token 被单独判断 |
| 主要指标 | informative K=50 时的 full coverage 和 recall，同时关注漏检和误报 |
| 防泄露措施 | 所有打分提交之后，评估才读取私有 token<br>Random 条件下，真实图像不参与打分 |
| 样本数 n | 取最大的一组：逐 token 校准的 10（开发）+ 20（验证）+ 20（复现） |

## 结果

### 稀疏字典排序

**回答**：只在已知图像时有效：验证集 full coverage 从 5/10 升到 9/10；改用 random surrogate 后从 5/10 降到 2/10。

**原理**：基线逐个判断 token；稀疏字典排序把基线排名前 2000 的候选当作"字典"，用 group lasso 求解"最少用几个候选 token 就能共同重构出观测到的梯度"，再按每个 token 分到的系数大小排序。真正出现过的 token 应该能用少数几个联合解释梯度。

开发集还测试了以下几种规则，作为对比：

- **硬谱打分**：只保留梯度最强的前 r 个奇异方向（r 取 6、8、12、16，或等于公开的 token 长度，或按奇异值的最大间隔自动选），在这个更小的子空间里计算距离，目的是去掉弱方向里的噪声。
- **软谱打分**：不直接截断，而是按奇异值大小给每个方向加权（权重为 s² / (s² + (c·s₁)²)，c 取 0.001、0.01、0.05），越弱的方向作用越小。
- **残差归一化**：把基线距离乘以 token 向量在去掉公开方向后剩余的范数比例。
- **投影融合**：分别只用 Q、K、V 一组梯度计算距离，再把这三个排名和基线排名融合（平均排名、中位排名、最好排名，以及 reciprocal rank fusion）。
- **多种子融合**：用 3 张不同的随机替代图分别打分，再融合排名。
- **自适应间隔截断**：不固定 K，而是在分数出现明显跳变的位置截断（相邻差值 ≥ 0.15 且比值 ≥ 2），找不到跳变时保留 100 个。

结果：

- **已知图像时明显改善**：开发集 full coverage 从 6/10 升到 10/10，验证集从 5/10 升到 9/10。
- 验证集上新覆盖了 4 个样本，没有丢失任何原本完整覆盖的样本。
- 开发集上最差的真实 token 排名从 1090 提升到 30。
- 验证集剩下的 1 个漏检不在最初的 Top-2000 字典中（排名 2746）；没有根据验证集扩大候选池。
- **用 random surrogate 时反而退化**：full coverage 从 5/10 降到 2/10，recall 从 0.933 降到 0.695。
- **其他规则都没有帮助**：硬谱和软谱打分、投影融合、3 个随机替代图融合，都没有提高固定预算下的覆盖；自适应间隔规则减少了候选数，但也丢掉了覆盖。
- 部分稀疏求解达到了迭代上限（已知图像下，开发集 10/10 收敛，验证集 7/10）；停止条件和 KKT 诊断保留在产物中。

| Image | Split | Baseline full coverage | Sparse full coverage | Baseline recall | Sparse recall |
|---|---|---:|---:|---:|---:|
| GT | development | 6/10 | 10/10 | – | – |
| GT | validation | 5/10 | 9/10 | 0.95833 | 0.99167 |
| Random | validation | 5/10 | 2/10 | 0.93333 | 0.69470 |

| Item | Setting |
|---|---|
| 开发集消融 | 10 个配对更新上的 40 个条件（已知图像和随机种子）<br>测试了硬谱和软谱打分、残差归一化、Q/K/V 打分融合、稀疏字典支撑，以及一个预先声明的自适应间隔截断 |
| 多种子融合 | 3 个替代图种子的结果只在读取参考之前融合 |
| 基线复现 | 现有基线打分的复现误差在 1e-14 以内 |
| 冻结参数 | 原 Top-2000 informative 字典<br>group lasso，λ/λmax = 1e-4<br>float64，最多 4000 次迭代，带符号系数<br>未激活的候选按原基线顺序补齐 |
| 验证集 | 10 个独立的 `eval` 图像组 |

**数值诊断（使用私有数据）**：用私有的前向和反向量重算梯度，攻击者拿不到这些数据，结论只用于诊断。

- 它精确复现了每个观测到的第一层 A 梯度。
- 用 float64 重新计算这些矩阵乘积后，10 个开发样本的所有真实 ID 都排在 informative 排名的前 6–16 位。
- 原生计算与重新计算之间的相对误差只有 0.00153–0.00188，但 span 的归属却变化很大。
- 答案末尾的弱信号不是普遍原因：被漏掉的 `ay` 系数并不小，而 Run 9 的末尾系数约为问题部分中位数的 0.017。

### 公开图像统计加权

**回答**：没有通过验证：开发集 full coverage 从 6/10 升到 8/10，验证集从 8/10 降到 5/10 和 6/10。

**原理**：图像私有时，基线只能用随机替代图来估计"视觉信息张成的方向"。这个方法改用 12 张公开示例图片，统计视觉输入最常出现的方向及其强度（二阶矩），对这些方向做软抑制：方向越强，压得越多（ridge 加权；ridge 为 0 时等同于把这些方向直接投影掉）。然后在加权后的空间里做方法一的稀疏字典排序。

- **Public-visual joint fit**：先去掉模板文本方向，再做上述视觉加权，最后做稀疏字典排序。
- **Random-surrogate joint fit**：对照组，仍用随机替代图去掉视觉方向，再做同样的稀疏字典排序。

结果：

- **开发集上有提升**：full coverage 从 6/10 升到 8/10，两种方法都是如此。
- **验证集上退化**：基线 8/10，两种新方法分别降到 5/10 和 6/10。
- 公开视觉方法在验证集上多覆盖了 run 8，但丢掉了 run 0、2、9，run 3 的 recall 也变差。
- 退化不是因为候选池漏掉了 token：每个验证样本的真实 token 都已在 Top-2000 候选池中。
- 退化也不是因为优化没收敛：4 个 recall 变差的公开视觉求解都满足收敛条件。总体收敛情况为 random 9/10，visual 7/10。
- 激活的候选数从开发集的中位数 10.5 增加到验证集的 170，说明可能存在随样本变化的干扰或模型失配，但不能证明这是唯一原因。

开发集（K=50）：

| Method | Full coverage | Recall |
|---|---:|---:|
| Baseline | 6/10 | 0.95089 |
| Random-surrogate joint fit | 8/10 | 0.96518 |
| Public-visual joint fit | 8/10 | 0.97232 |

验证集：

| Method | Full coverage@50 | Recall@50 | Full coverage@100 | Recall@100 |
|---|---:|---:|---:|---:|
| Baseline (random-surrogate span) | 8/10 | 0.98786 | 9/10 | 0.99286 |
| Random-surrogate joint fit | 5/10 | 0.92380 | 6/10 | 0.94523 |
| Public-visual joint fit | 6/10 | 0.93114 | 6/10 | 0.96341 |

| Item | Setting |
|---|---|
| 公开视觉数据 | 12 张 scikit-image 自带的示例图片，在投影掉已知模板后，提供独立的公开视觉二阶矩 |
| 开发阶段 | 两个阶段，测试了只用模板和随机的基线、视觉秩 64/256/512、ridge 系数 0.1 到 1e-7、硬投影对照，以及 group lasso 系数排序 |
| 范围调整 | 第二个 ridge 范围是在查看公开特征值谱和第一轮开发结果后加入的；两个阶段都没有接触验证集 |
| 候选池 | 所有联合拟合都使用原来 random surrogate 的 Top-2000 候选池 |
| 冻结参数 | Random-surrogate joint fit：λ/λmax = 0.01<br>Public-visual joint fit：视觉秩 256，相对 ridge 1e-7，联合 λ 比例 0.01<br>两者都用 float64，最多 4000 次 FISTA 迭代，未激活候选按原顺序补齐 |
| 验证集 | 10 个新的 eval client-0 图像组（offset 10–19，与稀疏字典的验证集不重叠） |

### 逐 token 校准

**回答**：只有局部提升：第一组验证集略有改善，随机复现时没有提升，recall 还略有下降。

**原理**：有些 token（比如常见词）不管是否出现在文本里，到梯度子空间的距离都偏小，容易被误选。这个方法用 48 个攻击者自己构造的公开样本（知道其中出现了哪些 token），估计每个 token "没有出现时"的距离分布（null 分布），再把私有样本上的距离换算成相对这个分布的 z 分数；z 越低，越说明这个 token 真的出现了。

在这个基础上测试了几种组合方式：

- **Null-rank mixture（`null_mix_0.5`）**：基线排名和 z 分数排名各占一半，取平均。
- **A-only / A/B logistic**：在公开样本上训练逻辑回归，预测 token 是否出现；A-only 只用 LoRA-A 的距离及其 z 分数，A/B 再加上 LoRA-B 的统计能量。
- **LoRA-B 统计能量**：token 向量经 LoRA-A 压缩后，落在 B 梯度中"无法由 B 权重解释的部分"上的能量。
- **Surface grouping**：去掉大小写和前导空格后字面相同的 token 视为一组，组内共享最好的排名。

结果：

- **开发集上有小幅提升**：`null_mix_0.5` 下 K=50 的 full coverage 保持 6/10，recall 从 0.95089 升到 0.96518；K=100 的 full coverage 从 6/10 升到 8/10，recall 从 0.95804 升到 0.97946。
- **第一组验证集有提升**：K=50 的 full coverage 从 17/20 升到 18/20，K=100 从 18/20 升到 19/20，没有样本的 recall 下降。
- **随机复现没有提升**：两种方法的 full coverage 都是 16/20，recall 还略有下降。
- **总体看 full coverage 会掩盖局部损失**：个别 token 的排名可能大幅变差，但不影响 full coverage 的计数。
- **LoRA-B 证据没有帮助**：只用 B 的打分很差，A/B 联合在 K=100 时更差。

| Method | Full coverage@50 | Recall@50 | Full coverage@100 | Recall@100 |
|---|---:|---:|---:|---:|
| Baseline | 33/40 | 0.98121 | 34/40 | 0.98671 |
| Null-rank mixture | 34/40 | 0.98462 | 35/40 | 0.98641 |
| A/B logistic | – | – | 34/40 | 0.98270 |
| Surface grouping | – | – | 33/40 | 0.97484 |

上表合并了两组共 40 张验证图像。各阶段的细节如下：

- **开发集**：
    - A-only 和 A/B logistic 在 K=50 时都是 6/10，recall 0.95804。
    - surface group 没有提高覆盖，反而降低了 recall。
    - 公开留出集（来自 4 张图像的 16 个配对样本，不是 16 张独立图像）：基线 K=50 为 9/16，rank mixture 和 `AB_0.01` 都是 16/16。
    - 第一层 B 输入的 sketch 秩为 8，即使只用投影后的公开模板也是 8；删除排名靠前的候选后仍为 8。所以无约束的 B-span 和单 token 删除检查都已饱和，但这不排除更结构化的 A/B 推断。
- **第一组验证集**（offset 20–39）：
    - K=50 时 recall 从 0.98429 升到 0.99143；K=100 时从 0.99143 升到 0.99500。
    - 个别 token 排名改善：Run 16 的 `ay` 从 83 升到 35，`image` 从 56 升到 28；Run 17 的 `ay` 从 184 升到 59。
    - K=100 下剩下的漏检是 Run 19 的 `ay`（215 → 104）。
    - A-only logistic 与 null-rank mixture 结果相同；A/B 在 K=100 时更差；大小写和空格分组会降低 recall。
    - 校准时没见过的 token 和样本组合：K=50 从 141/145 升到 143/145，K=100 从 143/145 升到 144/145。
- **随机复现**（20 张图像，种子 7391，offset 40 之后）：
    - recall 在 K=50 时从 0.97814 降到 0.97782，K=100 时从 0.98199 降到 0.97782。
    - K=50 时 run 13 改善、run 15 变差；K=100 时只有 run 15 变差。
    - Run 15 中 `What` 的排名从 13 跌到 486，尽管 `ay` 从 351 升到 146；它的 null-z 为 −1.9868，null rank 为 2037，与 z 截断无关。
    - Run 13 中 `ay` 从 1745 跌到 2847，而 `image` 从 56 升到 31。
- **对照**：只用词频时 full coverage 为 0；打乱梯度后基线的 full coverage 在第一组为 12/20，在复现组为 4/20。

| Item | Setting |
|---|---|
| 公开反事实样本 | 48 个：12 张 scikit-image 图片、自编的通用文本，以及等长的 token 替换对<br>前 8 张图像用于拟合；后 4 张图像和不同的模板作为公开留出集 |
| 测试内容 | A-only 与 A/B logistic、null-rank mixture、tokenizer surface group、公开词频对照、打乱梯度对照 |
| LoRA-B 的用法 | 取 B 梯度中与 LoRA-B 权重列空间正交的部分，作为统计能量特征；不把它当作 embedding span 筛选 |
| 带入验证的方法 | `null_mix_0.5`、`AB_0.01`、`group_AB_0.01`，以及基线、A-only 和 B-only 消融、两种对照 |
| `null_mix_0.5` | 对原始排名和 token 专属 null-z 排名取平均 |
| 第一组验证集 | 20 个新的 eval client-0 图像组，offset 20–39 |
| 随机复现 | 20 张图像，固定种子 7391，从 offset 40 之后抽取<br>选择时不看文本和分数，排除所有之前用过的图像，不重新调参 |

**样本多样性的限制**：

- 第一组验证集的 20 张图像只有 6 种不同的问答文本（重复次数为 14、2、1、1、1、1），有 13 个循环打乱的配对文本完全相同，削弱了打乱对照的效果。
- 随机复现组有 10 种不同的问答文本，最多重复 8 次，有 4 个打乱配对文本相同。

## 结论

- 三种方法都没有稳定超过现有筛选器，保留原来的筛选器。
- 稀疏字典排序在已知图像时确实有效，但不能作为通用的替代。
- 图像私有时，公开图像统计加权和 逐 token 校准都没有通过独立验证。
- 总体的 full coverage 会掩盖个别 token 排名变差的情况，评估时需要逐个样本检查。
- 这一阶段只涉及第一阶段，不声称句子恢复有任何改进。

## 局限

- 代码已提交于 `7be400e`；原始结果保存在本地 `outputs/`，不纳入 git。
- 每个验证集只有 10–20 张图像，部分验证集的问答文本重复度很高。
- 稀疏求解不是在所有样本上都收敛。
- B-span 和删除检查已经饱和，这不排除更结构化的 A/B 推断。

## 下一步

- 诊断第二阶段的 beam search，并实现一个忠于原论文的阈值解码器作为对照。

??? note "运行记录"
    - 所有 worker 只用 GPU 5，所有 watcher 都已退出，没有失败任务。
    - 稀疏字典：第一次尝试（路径不带 `_v2`）因为 CPU/GPU 张量放置失败，已保留；修复没有改动受害模型和研究设置。最终 CPU 回归 261 通过、3 跳过；Ruff 无报错；原研究的 summary 哈希没有变化，也没有替换原攻击的默认设置。
    - 公开图像统计加权：CPU 验证 263 通过、3 跳过；Ruff 无报错。新增的两个数学测试检查了 ridge 干扰消除的等价性和正交文本的保持。
    - 逐 token 校准：保留了失败的尝试——base 启动后在生成计划前退出；v2 遇到没有 tokenizer 对应的 padding embedding 行；v3 把 JSON 列表和 Python 元组做比较；v4 需要把加载的 B 梯度显式放到 GPU 上。v2 中已完成的公开特征和 6 个拟合模型被原样复用（字节一致）；v5 先验证了加载观测后的 A/B 打分和基线的精确复现。
    - 提交后的 informative 候选删除审计排除了禁用和 ambiguous token，1500 次删除后秩仍为 8。
    - 全部 50 个样本的打分产物和 3 份最终报告都通过了哈希校验。没有改动受害模型设置，没有 OOM。最终 CPU 回归 267 通过、3 跳过；Ruff 无报错。
