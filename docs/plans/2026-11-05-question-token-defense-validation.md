# DAGER 问题筛选验证

日期：2026-11-05。状态：六个 Stage-1 方向均已完成；没有完整筛选主方案通过门槛，P2–P4 按计划不运行。局部首词线索单独保留。本文把 [问题 token 改进思路](2026-11-05-question-token-defense-ideas.md) 中的六个方向变成可执行的验证矩阵。

## 目标与总体安排

目标是在相同候选预算下增加防御后的问题 token 覆盖，并区分梯度证据、语言先验和配额调整各自的贡献。每个方向先单独验证，负结果也完整记录；不把所有方案一次性叠加。

| Phase | Purpose | Data | Output | Advancement |
|---|---|---|---|---|
| P0 | 基线复现、公开校准与边界检查 | 原上传和攻击者自造样本 | 输入哈希、校准参数、公开审核 | 实现与协议检查通过 |
| P1 | 六个方向分别验证 | 原 SLAKE30 | 单独读出、固定预算补充、配对变化 | 不要求前一个方向有效，六个方向均测试 |
| P2 | 检查有效方法能否互补 | 同 SLAKE30 | 最多两个 2×2 对照 | 对应单方法通过预设门槛 |
| P3 | 冻结方法后独立确认 | 计划 20 个新图像组 | 冻结主方案与对照的候选结果 | 数据隔离审计和来源检查通过 |
| P4 | 检查候选收益能否变成序列收益 | P3 同一批样本 | Word Recall、ROUGE、EM | Joint FullCoverage 有明确增益 |

**六个思路都验证**指 P1 全覆盖，不是强制每个分支都跑昂贵的完整序列搜索。P2–P4 是有明确门槛的后续阶段；若六项均没有可用增益，负结果就是本轮结论，不临时增加参数或层网格。

主关注条件为 Random，GT 为机制和图像失配对照。Noise 与 Top-k 分开判断，不用混合平均掩盖某一防御的退化。None 用于检查新方法对干净梯度是否造成损害。

## 固定实验设置

| Item | Setting |
|---|---|
| Model | 原 SLAKE LLaVA-1.5-7B，F-L，Round 10，rank 32，alpha 64 |
| Precision | 原模型与原上传 bf16；矩阵评分 float64，LM 前向保持原模型精度 |
| Source | `outputs/diagnostics/federated_text_slake30_20261023/` |
| Existing B scores | `outputs/diagnostics/last_b_layer_sweep_20261104/`，复用 L31 down、L32 o 与原 prior，不重扫层 |
| Uploads | None、Noise ratio 0.01、Top-k 10%，每种 30 个原上传，共 90 份；不重捕获或重新扰动旧目标 |
| Views | 各上传配 GT 与 Random，共 180 个条件视图；两视图上传张量必须相同 |
| Surrogate | 沿用原 seed=0 随机像素图、原尺寸与处理方式；不挑选更接近真图的替代图 |
| Budget | informative 100，另保留原共同 9 个 ambiguous ID，实际候选数恒为 109 |
| Candidate ordering | 沿用特殊 ID 排除；数值同分按 token ID 稳定排序；禁止逐目标真值择层、择方法或补词 |
| Length knowledge | 仅使用 Observation 声明的问题和答案内容长度；不读取私有 loss mask 或参考文本打分 |
| Production code | 原攻击和模型不变，新评分走独立 evaluation 入口；没有 Git 提交 |
| Result page | 完成结果继续扩展 [DAGER 上传防御](../experiments/2026-10-23-dager-upload-defense.md) |

运行准备时从原计划解析模型实际路径，校验 metadata、权重指纹和公开 Observation 哈希；不要凭目录名推断版本。原 448 个 LoRA 上传张量覆盖所需模块，仍需逐个检查参数名、形状和可用性。

## P0：复现与公开校准

### 历史复现

- 复现原 informative 排名、共同 ambiguous ID、Baseline 109 个候选及原 B 的 80/20 补充。必须逐集合相等；已缓存 B 分数直接核验哈希，不改变符号、归一化和坐标。
- Question、Answer、Joint 的原候选指标重算后须与既有聚合一致。新增问题独有词指标不要求历史聚合已有，但不能改变历史指标定义。
- 分数重现时检查有限值、缩放不变性、零向量处理、dtype 和维度；异常保存 failed 状态并退出，不自动改 batch 或精度。

### 公开校准集

不使用 SLAKE 目标问题或答案拟合提示、正则和阈值。准备 12 个公开图片与人工编写问答的内容家族；图片只从已有攻击者公开示例库取，不从目标私有图片附近生成。

每个家族包含四个 QA 条件：原问答、同问题换答案、同答案换问题、同问题换一种长度相近的表述。替换词和完整 QA 都由攻击者预先编写，记录分词后长度和内容哈希。问题替换尽可能保留位置长度，以免只学习长度变化。

| Split | Families | Native gradients | Defense views | Use |
|---|---:|---:|---:|---|
| Fit | 8 | 32 | 96 | 只选 Masked ridge 正则，检查提示与数学实现 |
| Audit | 4 | 16 | 48 | 固定规则的公开审核，不据此反复改规则 |
| Total | 12 | 48 | 144 | 每个干净梯度派生 None、Noise、Top-k，无需重复反传 |

家族按固定序号划分，不能把同图或近重复问句随机分到两边。公开 QA 不必代表医学问答分布；领域失配是明确局限。若使用医学术语或医学提示，注明这是公开领域先验，不称为无领域知识。

公开捕获与目标使用同一模型和原生答案监督路径。Random 模拟中，评分端仅使用固定替代图，公开原图虽为攻击者自有，也不进入该分支的特征。新校准只选显式列出的一个正则参数，不训练额外分类器。

## P1：六个方向的独立验证

### E1：答案条件化的问题候选

**研究问题**：由当前上传得到的预测答案词片，是否比同预算、无答案线索的 LM 更能补回问题独有词？

固定 L31 down 的原能量排名，取前 5 个有效词片作为线索。特殊 ID 排除，其余规则不按目标效果改变；词片不拼成“真实答案”，也不使用参考文本纠错。

固定三个纯文本提示，每个最多贪婪展开 16 步，EOS 结束。线索以 JSON 字符串表示，提示明确它们可能不完整或错误。初始模板如下，公开准备阶段只允许修正格式或实现错误，不在目标结果上改写：

```text
The following are noisy candidate answer pieces for an image question: {pieces}.
Some pieces may be incorrect or incomplete. Write one possible question:

Possible answer pieces, possibly noisy: {pieces}.
A question about the image that could have this answer is:

Candidate answer fragments: {pieces}.
Generate a single image question consistent with some of these fragments:
```

每条路径平均各位置 next-token 概率，再对三个提示等权平均；同时保存实际生成 ID。该值只是操作性问题先验，不称严格的 token 出现概率。无内容生成不补采样、不悄悄换提示，报告退化率。

| Variant | Answer hints | Function |
|---|---|---|
| Q-blind | 用 `unknown` 占位 | 同模板、同生成预算的公开先验 |
| Q-matched | 当前上传的预测词片 | 主方案 |
| Q-wrong | 同防御下一份上传的预测词片 | 上传特异性对照 |
| Historical marginal | 原固定五模板 prior | 历史参照，生成预算不同，单独注明 |

Q-wrong 复用已完成的 donor 分数，禁止按真实答案挑选 donor。目标可能共享答案，错配对照不保证零命中。预测线索不含图像，因此每上传只生成一次，GT/Random 复用同一分数。

先各自取 100 个 informative 候选。再做完全同预算的三路补充：原 Baseline 先取 60 个，B 排名补 20 个不重复 token，问题 prior 补 20 个不重复 token；共同 ambiguous 最后加入。Matched、blind、wrong 使用相同的前 80 个，区别只在最后 20 个。另保留原 Baseline 与历史 B 80/20，不混作相同配额。

全部 None、Noise、Top-k，GT/Random 均报告。三种 prior 的同预算差值是主比较；仅高于历史 marginal 不足以归因于答案条件化。

### E2：Top-k 支持集上的公开拟合

**研究问题**：修正稀疏上传的观测模型，能否增加问题证据，而不是仅降低某个矩阵误差？

只在公开 metadata 声明 Top-k 的上传上启用，记录为防御适配；与原 defense-unaware 方法分列。公开子空间与当前图像条件一致：GT 使用声明的公开图，Random 使用原替代图。

| Variant | Public fitting | Candidate transform |
|---|---|---|
| Original span | 原完整坐标正交投影 | 原 span，历史复现 |
| Original energy | 原完整坐标正交投影 | 与 E3 相同的 K/V 能量 |
| Original row energy | 原完整坐标正交投影 | 每行归一化相关能量，作为严格同评分对照 |
| Post-mask energy | 原残差乘回非零掩码 | 每行同一 Post-mask 算子作用于候选 |
| Masked ridge energy | 保留坐标上带正则拟合 | 每行相同支持集、相同 ridge 算子作用于候选 |

Original energy 与 Original row energy 是必要对照，否则无法区分“换能量评分”“改变行归一化”与“修正掩码”的贡献。四种能量方法固定仅用 K/V 等权，Q 作为诊断另记，不增加新的私有目标调参。

令 $U$ 为列正交的公开基，$S_i$ 是第 i 行的坐标选择算子，$D_i=S_iU$，$y_i=S_ig_i$。定义

\[
H_i=I-D_i(D_i^\top D_i+\lambda_i I)^{-1}D_i^\top,
\quad r_i=H_iy_i,\quad z_{i,w}=H_iS_ix_w.
\]

Masked ridge 的打分为各非退化行的归一化相关能量均值：

\[
s(w)=\operatorname{mean}_i
\frac{(r_i^\top z_{i,w})^2}{\|r_i\|_2^2\|z_{i,w}\|_2^2}.
\]

零范数行单独记录并排除；若整个模块没有有效行，保存该方法失败，不用 Baseline 伪装该方法已完成。固定共同 ambiguous，不把这些公共重叠词算作新阳性。

正则尺度取 $\lambda_i=\rho\|D_i\|_F^2/\max(1,|\Omega_i|)$。只在公开 Fit 比较 $\rho\in\{0.01,0.1,1\}$，按 Random 的 Question-exclusive recall、Question FullCoverage、较大正则依次打破平局，冻结一个值后使用 Audit 与目标。若 Fit 中无非空问题独有集合，直接报告校准不适用，不改为在目标选择正则。公开 Audit 的通过条件为 Top-k Random 的问题独有词召回不低于 Original row energy、答案召回下降不超过 0.02；未通过仍可完成目标上的预设负结果诊断，但不据此进入 P2 或宣称公开审核通过。

Post-mask 对照使用 $T_i=S_iQ$，$r_i=T_ig_i$、$z_{i,w}=T_ix_w$ 及相同相关能量公式；它不等同于 ridge。Original row energy 使用 $T_i=Q$ 和同一行相关能量公式。Original energy 使用完整 $Q$ 和 E3 的矩阵能量公式。主归因比较为 Masked ridge、Post-mask、Original row energy，不把其他评分差异归为掩码。

必须报告每行支持数、限制基秩、条件数代理、残差范数及数值稳定性。公开秩可能大于观测坐标数，无正则拟合可吸收全部私有信号；这是主要失败假设，不隐藏零残差。支持集的缺失依赖梯度大小，不宣称无偏补全。

每个新方法单独取 100 个 informative 候选，再测试“新问题评分 80、固定 B 补 20”。全部 30 个 Top-k 上传、两种图像，共 60 个视图；None 可作数学退化检查，Noise 不伪装成适用结果。

### E3：首层输入能量

**研究问题**：在相同公开消除下，保留梯度幅度、区分 K/V 能否优于原 span？

按 idea 文档公式，分别计算 K-energy、V-energy、KV-energy、Q-energy、QKV-energy。联合方法先对模块作 Frobenius 归一化，再等权平均；固定处理顺序，不以私有成绩调整权重。K/V 的模块平均是本方向主方案，另外四项均报告。

所有方法共用原首层 RMSNorm 词向量、公开基、特殊 ID 与 ambiguous；单独 100 个 informative、以及“新评分 80、固定 B 补 20”两种候选规则。共 90 份上传、180 个视图，不重捕获。

对照包含原 span、E1 的 Q-blind、同防御循环错配的能量分数。保存完整词表分数，报告与原 span 的排名相关及 Top-100 交并集；若平坦谱使两者几乎相同，这也是可解释的负结果。

先执行该分支是因为便宜，不改变 E1 在研究优先级上的第一位。理想等方噪声地板减去同一个常数不改变排序，不把这一步添加成伪改进。

### E4：实际前缀下的浅层 A

**研究问题**：使用真实模块坐标与完整因果前缀，是否能识别问题首词？这一步不冒充完整问题 token 筛选。

固定六个模块：首层 o、gate、up、down 的 LoRA-A，以及第二个 block 的 K、V LoRA-A。o 的输入是多头注意力合并后的实际向量；down 的输入是实际 SwiGLU 激活。特征必须由当前有效权重重放，包含 LoRA。

GT 使用 Observation 的公开图像前缀；Random 使用固定替代图。未知问题之后的模板和 padding 不进入深层公开消除基。首层 gate/up 的输入相同、梯度不同，分别报告，不当成独立输入证据。

为控制成本，首词提议池固定为同一个 text-only LM 在 `USER: ` 与换行分别分词后拼接的前缀下，next-token 概率最高的 256 个有效 ID；不放入答案线索，不按样本真值补入首词。沿用特殊 ID 排除，公共 ambiguous 的处理单独记录。所有模块及首层 K/V 对照使用完全相同提议池。记录真实首词落入池中的比例；池外按未命中计，不能只报告池内正确率。

各模块对候选池的实际输入做公开前缀消除，先用同一能量公式评分；原首层 span 是对照。只报告首词 Top-1、Top-5、Top-20 命中和运行开销。原 ambiguous 中的首词单独报告，informative 子集注明有效 N，防止公共保留词制造虚假提升。

原 30 QA、三种上传、两种视图全部测试。Random 若上下文完全相同可复用特征缓存，GT 按公开图像分别计算。只有首词分支有新证据才考虑未来的后续前缀；本轮不扫描所有位置或以真实问题前缀推进搜索。

### E5：配额与 B 共识

**研究问题**：问题损失是否主要来自名额挤占？两个既有 B 候选是否互补？

先对 L31 down、L32 o 各自的排序、候选交并集和新增命中做离线分析。固定 ensemble 为二者有效词表百分位排名的算术平均，不增加第三个层或择样本最优。

| B source | Answer-channel quota |
|---|---|
| L31 down | 8、20、动态规则 |
| L31 down 与 L32 o 的固定排名平均 | 8、20、动态规则 |

动态规则保留原提议：$K_A=\min(18,\max(8,3L_a))$，仅使用声明的内容长度。每行先取 Baseline 的 $100-K_A$ 个，再从对应 B 排名补齐到 100；追加共同 9 个 ambiguous。固定 20 的单 B 必须复现原 80/20。

共六个策略，90 上传、180 视图，全部复用已有分数。额外画 K=5、10、20、40 的答案候选召回曲线；它是预算诊断，不能将某条目标真值曲线上最好的 K 当作实际策略。

结果以 Question、Answer、Joint 的配对得失判断。问题回升但新增词全部来自原排名 81–100，只支持预算恢复，不能说提取了原候选之外的新问题证据。预设不把 E5 单独的小幅问题增益作为端到端推进理由。

### E6：B 前向探针及输入坐标对照

**研究问题**：外部建议的前向特征左乘 B，是否有超出一般几何偏好的经验信号？输入坐标正确的替代能否更合理？

固定首层 V 的 LoRA-B，不新扫层。令首层文本输入为 $x_w$，V 有效权重为 $W_V^{eff}$，$u_w=W_V^{eff}x_w$，$v_w=A_Vx_w$。

| Variant | Formula | Interpretation |
|---|---|---|
| Forward-left | $\|G_B^\top u_w\|^2/(\|G_B\|_F^2\|u_w\|^2)$ | 原建议的可计算启发式，没有反向对齐保证 |
| Permuted-left | 对 $u_w$ 作固定随机带符号坐标置换后用同式 | 保持向量范数、破坏特定坐标对应的几何对照 |
| Head-left | 同一 V 的 B，使用原输出词方向 | 检查换特征本身的收益 |
| Input-right | $\|G_Bv_w\|^2/(\|G_B\|_F^2\|v_w\|^2)$ | B 输入坐标匹配的压缩能量，不是 span 成员证明 |

带符号置换只用一个固定种子，先记录映射哈希，不在目标上挑随机映射。每个评分配同防御循环错配上传。None、Noise、Top-k 和两种候选视图均测；分数不读图像，视图差别只来自共用的候选排除规则。

单独 Top-100 为本方向主比较。原第一层 B 压缩证据在其他 rank 与无防御条件已有不稳定结果，必须在记录中交代，不能称从未测试过。若 Forward-left 不优于置换或错配，只支持一般词向量偏好；若 Input-right 有收益，也只能归于统计能量，不能补写不存在的反向对齐定理。

本轮不增加全词表逐词反向 Jacobian。E6 已检验原建议的直接版本及一个坐标正确的简单替代；更复杂的反事实回放须另定问题和预算。

## 指标、控制与选择规则

### 候选层面的指标

| Scope | Metrics | Notes |
|---|---|---|
| Question | Question recall、Question-exclusive recall、Question FullCoverage | 问题独有集合为 Q∖A；空集合不参与均值，报告有效 N |
| Answer | Answer recall、Answer FullCoverage | 防止问题收益来自牺牲答案 |
| Joint | Joint candidate recall、Joint FullCoverage | 按 Q∪A 去重计算，不能平均 Q/A 召回 |
| Ranking | Informative AP、真实 token 排名分位数 | 仅完整词表排序计算；融合和 E4 短名单不伪造全词表 AP |
| Cost | 前向次数、词表评分数、墙钟、峰值显存 | 缓存、校准与目标评分分别记录 |

E4 仅使用其首词专属指标，不与完整问题召回列成同一口径。所有方法分别报告 None、Noise、Top-k，GT、Random，以及原短中长三桶。候选集合均先保存，再由 evaluation 读参考；评分器不能接收真值、参考前缀或原数据标识。

报告逐图配对均值差、变好/变差/不变的样本数和按图像组重采样的 95% 区间；固定 10000 次 bootstrap、seed=20261105。原 30 条是反复使用的开发集，不对多方法中的最好 p 值作显著性宣传。

### 推进门槛

以下是实验推进的工程门槛，不是理论阈值或统计显著性的替代。必须在首份目标新分数出现前写入机器可读计划。

- E1–E3 或 E6 若要成为新增问题证据候选：Random 的 Question-exclusive recall 相对对应等预算对照至少提高 0.05，匹配上传相对错配方向一致；Question FullCoverage 不下降。
- 进入 P2 的完整候选方案，相对原 B 80/20 的 Answer recall 下降不超过 0.02，Joint FullCoverage 不下降。未达门槛仍完整报告六个方向结果，不换指标找赢家。
- E4 是首词诊断：informative 首词 Top-5 命中相对同池首层对照至少多 3/30，且匹配优于错配；通过只说明值得后续前缀研究，本轮不直接把它作为完整筛选方案。
- E5 若只是恢复原候选里的问题词，标记为预算收益；进入组合前须满足相同答案非劣门槛，不因问题 recall 单独涨一点便推进。
- 每种防御最多保留一个主方案进入 P3。先按公开 Audit 是否通过，之后在旧开发集按 Joint FullCoverage、Question-exclusive recall、Answer recall、较低成本的顺序排序。该选择属于开发选择，不称独立验证。

正则只由公开 Fit 决定，私有开发集只选择整套已冻结方法，不重新拟合提示、参数、词典或线索数量。P3 不再选择赢家，所有冻结主方案及失败都报告。

## P2：小规模组合消融

不做六方法的全组合。只在相应单方向通过后运行下面两个固定的 2×2：

| Defense | Factor A | Factor B | Fixed candidate budget |
|---|---|---|---|
| Noise | 原 Baseline 排名或 KV-energy 排名 | Q-blind 或 Q-matched | A 取 60、L31 down 补 20、B 补 20，另加共同 9 |
| Top-k | 原 Baseline 排名或 Masked ridge 排名 | Q-blind 或 Q-matched | 同上 |

若某因子未通过单方法门槛，则不称为“有效组件”，对应组合不运行并注明原因。Q-wrong 保留为匹配方案的控制，不计为第三个优化因子。E5 的动态配额不与这两个实验同时变化，以免归因混淆。

## P3：独立确认

目标为 20 个新的 SLAKE eval 图像组，用与当前受害快照一致的原数据划分；不重训。创建前审计本轮与以往文本调参使用过的全部图像组、像素重复及训练隔离，按固定种子选取，不按问答内容或方法成绩筛样本。新池自然长度分布如实报告，不强求每桶一样多。

样本清单由数据准备端冻结，评分端只拿匿名 Observation。若同一 eval 划分无法提供 20 个未用于文本方法开发的组，报告可用数和重叠，停止“独立确认”这一步；不静默改 split、模型或把旧样本称新样本。图像隔离不保证问答文本从未在训练中出现，需另报告文本重叠。

每个新样本原生捕获一次干净梯度，再由同一干净张量派生三种上传，共 20 次目标反传、60 份更新；GT/Random 共用相同更新。新 Noise 的种子在捕获前固定，不能比较后选择一份“更有利的噪声”。

None 只测冻结主方案是否损害干净条件。Noise、Top-k 各比较原 Baseline、原 B 80/20、各自唯一冻结主方案和对应必要先验/错配对照。全部预测提交后统一评价，确认集结果不得反过来改主方法。

## P4：序列恢复

只有新确认集上的主方案满足前述候选门槛、且 Joint FullCoverage 比原 B 80/20 至少多 2/20，才进入该防御下的完整恢复。若没有达到，候选诊断已经足以形成结论，不强制跑序列搜索。

每个通过的防御，比较原 B 80/20 候选和新候选；同一批 20 个样本、GT/Random、原 beam 16、LM prior 0.01、rerank=0、评分上限 200000、3600 秒上限。先取冻结顺序前 4 个做工程 pilot，之后用同一配置续完 20 个，前四个计入总数、不重复运行。不按真值 FullCoverage 选择哪些样本跑搜索。

最多两种防御、两种方法、两种图像、20 样本，即最多 160 次搜索；是否运行及预计耗时由首批实测给出，不凭先前矩阵分析的分钟数估计束搜索。超时保留为预算内未恢复，异常明确保存并退出，不自动改搜索宽度或精度。

| Defense | Method | Image | N | Word Recall | ROUGE-1 | ROUGE-2 | ROUGE-L | Question EM | Answer EM | Joint EM |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pending | Pending | Pending | — | — | — | — | — | — | — | — |

两个方法使用同一第二阶段评分，仅改变明确声明的第一阶段候选。因此提升可归于候选改进，失败也可能说明第二阶段对防御不鲁棒；不能把候选覆盖当序列结果。

## 执行、产物与完成条件

建议执行顺序：P0 → E3 与 E5 的便宜检查 → E1 → E2 → E4 → E6 → P2 → 有门槛的 P3/P4。执行顺序按成本安排，不改变 idea 文档的成功可能性排序。E1 失败不取消其余五项。

实现集中在独立的 `evaluation/question_filter_study.py`、`question_filter_public.py`、`question_filter_math.py`、`question_filter_evaluate.py` 和 `tests/test_question_filter.py`，不改生产攻击。数学、候选预算和延迟评价等 9 项新测试与既有 B 测试共 29 项通过，Ruff 通过。

根目录建议为 `outputs/diagnostics/question_filter_defense_20261105/`；若已有目录则使用新的明确后缀，禁止覆盖。目录分为 prepare、public、scores、candidates、evaluation、report、monitor，分数与候选的完成标志和哈希是评价入口的硬门槛。

- 每个方法写出完整冻结配置、所有输入来源、模型/Observation 哈希、源码快照、分数和候选 safetensors；模型、校准、GPU 评分与 CPU 评价分阶段记录。
- 新生成问题属于攻击产物，和逐目标线索、私有参考、样本标识一起留在 outputs，不发布到 docs。只复制聚合 JSON/CSV 和约 2400px 宽的曲线、条形图；标明未提交。
- 拟发布三张图：各方法的问题独有词召回；Question/Answer/Joint FullCoverage；配额导致的问答得失。E4 首词诊断单独出表，不把它混成完整问题效果。
- 单 GPU 为默认，启动前检查本机与 WN-A100 的真实进程和占用，本项目合计最多两张空闲物理卡；显式绑定 UUID。长期任务冻结代码、detached 启动并配退出型 watcher，十分钟状态记录，首份产物核验后才继续。
- 不共享忙卡、不停止他人进程、不修复驱动或 NAS；无可用资源时记录等待，不谎称正在计算。模型与环境仍在本机 NAS 准备，远端仅执行。
- 所有异常保存状态后退出，不自动 OOM 重试、减小实验 batch、切换 dtype 或替换方法；已有不完整产物保留，修复必须记录新阶段和等价性边界。

完成条件为六项 P1 各得到可复核结果或明确的实现/适用性结论，P2–P4 各有通过、未通过或资源/数据不足的明确状态，统一实验记录、来源、图表、索引和 handoff 均更新。不以“有一个方向有效”提前结束其他已列入 P1 的验证，不提交 Git。

## 进度清单

- [x] 六个方向的矩阵、对照、指标、门槛和资源规则已落盘。
- [x] P0：输入、公开校准、数学测试、历史复现与源码冻结完成。
- [x] E1：答案条件化、同模板无条件和错配对照。
- [x] E2：支持集正则拟合、Post-mask 与能量对照。
- [x] E3：首层 Q/K/V 能量及固定联合。
- [x] E4：实际前缀六模块的首词诊断。
- [x] E5：两个 B 来源与三种配额。
- [x] E6：前向左探针、置换、输出词与输入右探针对照。
- [x] P2：没有完整筛选主方案通过门槛，明确不运行组合。
- [x] P3：未达到推进门槛，不采集新样本或声称独立确认。
- [x] P4：联合完整覆盖未增加，不运行序列搜索，无新 ROUGE 或 EM。
- [x] 结果已并入统一实验记录，聚合与独立复核完成，远端进程及 GPU 退出已核验。

## 执行记录

- 用户本轮明确授权开始实验，先前关于只设计或执行的范围问题已解决，不再等待确认；仍不提交 Git。
- 输出 `outputs/diagnostics/question_filter_defense_20261105/`，base commit `2fd25b6`；冻结快照 `source-question-v1`，源码 SHA256 `68898372ed8d30275e6f52aae19d28c67e9f3d91e3d21e7819ae7ab2df908ada`。现有 `gia` 环境保持不变，代码与环境准备在本机 NAS。
- 启动前 WN-A100 GPU 2 与 7 均为 4 MiB、利用率零，本项目无在运行 GPU 任务；控制器最终只绑定 GPU 2，UUID `GPU-8cb10863-d4c8-528f-3add-2632ae9cb39d`。controller 2518、首个 public worker 2525、watcher 2530，未占用第二张卡。
- 启动命令：`ssh zx-a2 'cd /home/zx/nas/GitRepos/giavlm && bash outputs/diagnostics/question_filter_defense_20261105/setup/launch.sh'`。状态在 `controller.json`，阶段日志在 `setup/`；watcher 每十分钟记进度、每三十秒记显存，完成、失败或控制器死亡时退出。
- 六项评分和候选全部保存后才启动目标评价；原生 448 张量逐值对照与首份词表分数检查均是硬门槛。首份原生和词表评分审计均已通过，最终状态以下方完成记录为准。
- 中间阶段：公开48次捕获及144个防御视图完成，原生首份448张量逐值一致，180份原上传全文件哈希通过。正则由公开 Fit 选为0.01，公开 Audit 未通过；保持冻结值完成目标诊断，不把它作为已验证改进。E1 的90份匹配先验及一份同模板 blind prior 已保存，目标真值尚未读取；矩阵阶段已进入，首份分数审计仍待核验。


## 完成记录

- 远端 controller2518、public2525、prior2591、matrix2629、context2738、candidates2768、evaluate2782 和 watcher2530 均已结束，六阶段 returncode=0；总墙钟1311.05秒。最终 SSH 核验 GPU2 为4MiB、利用率0，本轮仅用一张物理卡，没有OOM、重试或设置变更。
- 公开校准的48个逻辑样本另有1次原生捕获等价性复核；全部180份原上传完整哈希通过。原基线分数最大差7.55e-15，10380组候选均为109；811个提交文件、72项历史指标、83040项独立指标及2768项聚合复核通过。
- 噪声下Q blind有明显部分候选收益，预测答案条件化未超过同预算先验；首层up和第二层K仅通过首词工程门槛。全部完整筛选主方案未通过，next-phases.json记录P2–P4不运行，不能把它们写为完成了新样本或序列实验。
- 结果与三幅聚合图见 [[dager-upload-defense]] 的“问题候选六方向”；完整条件、分桶、首词和配对结果在 docs/experiments/data/federated-text-pilot/question-filter-*。评分代码四个文件与68898372快照不变，报告和复核代码另列哈希；全部未提交Git。
- 最终53项定向与索引测试、Ruff、21份记录及本地产物、MkDocs严格构建、23个发布/代码哈希与完整冻结指纹通过，finalized.json记录完成。没有待运行任务，不重复启动。
