# 本项目与原版 DAGER 的实现对比

本项目的 `dager_adapted` 保留了 DAGER 的核心思路：第一层筛选 token，第二层验证序列前缀。但它不是原论文算法的直接移植。影响恢复能力最大的变化是低秩 LoRA 梯度、公开图像与长度假设、公开输入子空间投影，以及用有限宽度的累计分数搜索替代原版的阈值扩展。

本文记录 2026-10-02 的代码核对结果，区分论文条件、参考代码实际行为和本项目已有实验结论。

- 本项目核对版本：`187da5c`。
- 本地原版参考仓库版本：`5f8e306223ad35f3b8ce61b35f1ff60158207310`。
- 论文：Petrov et al., *DAGER: Exact Gradient Inversion for Large Language Models*, NeurIPS 2024，[arXiv v2](https://arxiv.org/html/2405.15586v2)。
- 官方代码：[insait-institute/dager-gradient-inversion](https://github.com/insait-institute/dager-gradient-inversion)。
- 原版以实验脚本调用的 `attack.py` 为主要比较对象，同时核对 `attack_new.py` 的噪声攻击分支。

本文的本地代码链接相对于 `attacks/analytic/`。`dager-gradient-inversion/` 是被忽略的只读参考 checkout；没有该目录的副本可通过上述官方仓库及提交定位原版代码。

## 实现范围概览

| 比较项 | 原论文及原版实现 | 本项目 dager_adapted |
|---|---|---|
| 攻击对象 | 纯文本 Transformer，包含 decoder 和 encoder | LLaVA 的 LLaMA decoder，VQA 文本 |
| 私有数据 | 一个 batch 中的文本序列 | 单样本的问题和答案，或仅答案 |
| 已知信息 | 不要求每条序列真实长度 | 必须公开图像和文本长度，可额外公开问题 |
| 梯度来源 | 完整投影权重梯度，另有 LoRA 扩展 | 前两层选定 Q/K/V 的 LoRA-A 梯度用于建子空间 |
| 默认子空间 | 原始梯度空间 | 去除公开输入方向后的空间 |
| token 检测 | 按位置或全局词表做阈值筛选 | 全局词表，阈值或 top-k，另保留歧义 token |
| 序列搜索 | 阈值扩展，另有近似候选保留机制 | 固定宽度 beam，累计第二层距离 |
| 长度与结束位置 | 通过候选位置、EOS 或无法继续扩展处理 | 真实长度已知，固定 EOS/PAD，只填内容位置 |
| 最终梯度匹配 | 标准搜索不做完整梯度重算 | 可选完整上传梯度重排，默认关闭 |
| 扩展功能 | FedAvg、encoder、噪声梯度等分支 | 当前均未迁移 |
| 恢复结论 | 在相应秩条件下讨论精确恢复 | 明确不承诺精确恢复 |

本项目原名 [`dager`](dager.py) 仍然返回 `not_implemented`；实际可运行的入口是 [`dager_adapted`](dager_adapted.py)。二者不能混称为原论文复现。

## LoRA 梯度与恢复条件

LoRA 不是本项目新增的支持。原论文附录 B.4 已讨论 LoRA，并指出零初始化的一个 LoRA 因子会导致另一个因子的梯度退化。原版实际 LoRA 脚本使用经过训练的 LLaMA-3.1-8B、`r=256`，只对 `q_proj` 注入 LoRA。原版 README 仍写 GPT-2，但实际脚本和模型包装代码对应的是 LLaMA 路径。

来源：[原版 LoRA 脚本](../../dager-gradient-inversion/scripts/lora.sh)、[原版 ModelWrapper](../../dager-gradient-inversion/utils/models.py)，以及论文附录 B.4。

按 PyTorch 权重方向记号，设投影输入为 `X`，反向信号为 `D`，完整权重梯度为：

```text
G_W = Dᵀ X
```

对于 `W = W₀ + s B A`，其中 `A` 的形状为 `r × d`、`s` 为 LoRA 缩放系数，有：

```text
G_A = s Bᵀ Dᵀ X
G_B = s Dᵀ X Aᵀ
```

因此，LoRA-A 梯度的行空间位于输入的 `d` 维空间中，可以直接用于检测输入向量；LoRA-B 梯度不能不加推导地替代它。论文使用的矩阵方向及 A/B 记号应按维度对应，不能把转置或记号差异误判成算法差异。

但一般只能推出：

```text
rowspan(G_A) ⊆ rowspan(X)
```

要让真实输入也属于观测梯度空间，需要反向系数保留足够的输入方向。“梯度低秩”本身并不够。原论文完整权重分析涉及反向系数满秩以及总 token 数小于隐藏维度的条件；LoRA 扩展进一步要求相应满秩条件，并给出 `b < r` 的假设。不能只沿用完整权重场景中的隐藏维度条件。

本项目默认 LoRA rank 为 8。单个投影最多提供 8 个方向，联合 Q/K/V 后每层最多提供 24 个方向，比原版 rank 256 的条件严格得多。这里的 24 是代数上的方向数上界，不代表一定获得 24 个有用、独立且准确的私有输入方向。

项目专门包含一个反例测试，证明真实输入可以全部不落在较小的观测梯度空间里。当前支持检查只确认必要梯度存在、形状正确、每层所选梯度不全为零，不会证明精确恢复所需的秩条件已经成立。

当初始化导致必要 A 梯度为零时，项目返回 `not_applicable`，不会自动换用 B 梯度、完整权重梯度或改变训练状态。检查基于实际梯度，而不是简单用训练轮数判断。

来源：[数学与接口测试](../../tests/test_dager.py) 中的 `test_linear_and_lora_gradient_identity_and_initial_degeneracy`、`test_low_rank_alone_does_not_imply_true_token_membership`，以及 [dager_support](dager_adapted.py)。

## 联合 QKV 子空间与受害者微调范围

原论文第 3.2 节说明后续不失一般性地使用 Q 展开推导：当相应满秩条件成立时，可以在 Q、K、V 中任选一个，而不是必须拼接三个。若三个投影都完整保留输入空间，其梯度行空间相同，拼接不增加新的输入方向。本项目联合三个 LoRA-A 梯度，是尝试补足低 LoRA rank 下不同投影保留的互补方向；它不是原论文要求，也没有一般性的性能提升保证。

原版 LLaMA 路径主要读取 Q 投影梯度；GPT-2 使用融合 QKV 的 `c_attn` 权重，因此不能笼统地说原版所有架构都只使用 Q。

本项目通过参数名称定位前两层的以下上传张量：

```text
self_attn.{q,k,v}_proj.lora_A.*.weight
```

默认分别用 Frobenius 范数归一化三个梯度，然后纵向拼接，做一次 SVD。它得到的是三个行空间共同张成的空间，不是要求候选同时通过 Q、K、V 三次独立检查，也不是三个检测器投票。

归一化发生在公开方向投影之前，避免把几乎完全属于公开空间的梯度所留下的微小数值残差重新放大。`projections` 可以选择 `qkv`，也可以只用 `q`、`k` 或 `v`。

原版主要通过参数列表下标和模型专用 `layer_ids` 定位梯度；本项目使用参数名并检查张量形状。两层分别建立子空间，不把不同层的梯度混成一个空间。

另一个不能忽略的区别是受害者自身的微调范围。本项目给语言模型中除输出头之外的 Linear 层注入 LoRA，包含 attention 和 MLP；原版上述 LoRA 脚本只训练 `q_proj`。F-CL 还会同时训练 connector。这些差异会改变前向状态和反向信号，不能只看攻击读取了哪些梯度。

来源：[gradient_groups](dager_adapted.py)、[SpanFilter.build](dager_subspace.py)、[本项目 configure_training](../../core/vlm_wrapper.py)、[原版 ModelWrapper](../../dager-gradient-inversion/utils/models.py)。

## 公开输入子空间投影

`public_residual` 是本项目最实质性的算法扩展之一。原版直接检测候选向量与梯度空间的距离；本项目默认先构造公开输入空间，移除图像、模板等已知输入方向，再检测剩余部分。

令 `P` 为公开输入空间的正交投影，`R = I - P`。项目实际使用的矩阵可写成：

```text
G̃ = stack(G_AQ / ‖G_AQ‖F,
           G_AK / ‖G_AK‖F,
           G_AV / ‖G_AV‖F) R
```

对候选向量 `x`，使用归一化距离：

```text
d(x) = ‖xR - proj_rowspan(G̃)(xR)‖₂ / ‖xR‖₂
```

若梯度包含未知反向系数乘以公开输入的项，投影可以消去该项，无须知道那些反向系数。代码通过正交基实现投影残差，不显式构造完整的 `d × d` 投影矩阵。

两层使用的公开空间不同：

- 第一层使用所有已知位置的输入，包括图像特征、模板、已知问题，以及结构性 EOS/PAD。
- 第二层只使用第一个未知 token 之前的因果前缀。

未知问题之后的 `ASSISTANT:`、EOS、PAD 虽然 token ID 已知，经过第一层 attention 后却依赖未知问题，不能当成已知隐藏特征投影掉。当前实现区分了“已知 token ID”和“已知上下文特征”。

据此可以推断：即使私有内容只有 14 个 token、联合梯度有 24 个方向，第二层仍可能存在大量依赖私有上下文的其他输入方向，不能直接用“14 小于 24”推导序列一定可恢复。这是对代码和梯度结构的分析，不是已经独立验证的失败归因。

`mode=raw` 只关闭公开方向投影，仍保留 LoRA、QKV 拼接、数值设置和 beam search 等改动，不等于恢复了原版 DAGER。

来源：[SpanFilter](dager_subspace.py)、[LlavaTextView](../../core/adapters/llava.py)。

## 公开知识与 VQA 训练目标

当前实现只接受两种知识条件：

- `image_known`：图像公开，问题和答案私有。
- `image_question_known`：图像与问题公开，只恢复答案。

两者都必须设置 `token_lengths_known=true`。因此本项目没有实现未知长度推断，也没有实现图像与文本联合恢复。最终 `Reconstruction.images=None`，图像只是固定的公开输入；公开问题也不会参与优化或恢复评分。

原版主要处理文本分类或普通 next-token prediction；本项目采用固定布局：

```text
USER: + 图像特征 + 换行 + 固定长度问题槽 + ASSISTANT: + 答案前缀
```

损失只作用于答案及其 EOS，问题和图像通过上下文影响答案损失。问题槽中的 PAD 仍保留在 attention 上下文中；原版文本路径则显式传递由 PAD 生成的 attention mask。

这些差异改变梯度系数、有效秩和第二层上下文特征，并非只是在输入开头附加一张图片。固定槽、EOS/PAD 和 loss mask 必须与项目受害者协议一起理解，不能替换成另一种常规聊天模板后仍认为观测梯度相同。

答案内容后包含 EOS 监督，所以最后一个内容 token 可作为预测 EOS 的输入。不能机械套用普通 next-token 序列中“最后一个 token 只作为标签”的分析。

来源：[llava_input_pieces](../../core/adapters/llava.py)、[HFAdapter.target_logits](../../core/adapters/hf.py)、[VLMAdapter.forward](../../core/vlm_wrapper.py)。

## 第一阶段 token 筛选

原版对 GPT-2/BERT 枚举 token 与位置组合；LLaMA 首层输入不含绝对位置嵌入，因此只扫描一次词表，得到全局 token 集合。本项目沿用后者，检测首层 RMSNorm 后的 embedding，而不是未经归一化的裸 embedding。

本项目第一阶段得到的是问题与答案私有 token 的并集，不判断 token 属于哪个字段，不恢复顺序，也不恢复重复次数。后续每个私有槽使用同一个候选集合。

| 规则 | 本项目实际行为 |
|---|---|
| `threshold` | 保留距离不大于 `token_threshold` 的非歧义 token，再按上限截断 |
| `topk` | 直接取距离最小的非歧义 token |
| `max_candidates` | 默认 256，两种模式都生效 |
| 公开空间歧义 token | 额外保留，不占上述上限 |
| 特殊 token 与无效 embedding 行 | 排除出内容候选 |

因此，`max_candidates=256` 不意味着最终候选总数最多 256。截断情况会记录在 provenance 中。

被公开投影几乎完全消掉的向量被标记为 `ambiguous`，距离设为零，其 token ID 被保留。该 token 可能重复出现在私有文本中，但零残差不能证明它确实出现。这个处理避免误删，同时引入缺乏第一阶段检测证据的搜索候选。第二阶段遇到同类歧义特征时也计数，并贡献零距离。

原版 `get_top_B_in_span()` 虽然名字含 `top_B`，其函数实际只是返回所有通过阈值的 token 并排序，没有在函数内截成 B 个；截断由其他调用路径决定。

本项目还排除了 LLaVA 为矩阵对齐而补出的 embedding 行。本地模型有 32,064 行 embedding，但 tokenizer 只有 32,002 个 ID。旧 raw 消融的 top-50 曾有 49 个来自无效行；当前代码已经修复，旧结果不能作为有效对比证据。词表扫描计数仍可包含被扫描后排除的行，不能将扫描行数等同于有效内容候选数。

来源：[filter_tokens 与 forbidden_token_ids](dager_adapted.py)、[select_tokens](dager_subspace.py)、[原版 get_top_B_in_span](../../dager-gradient-inversion/utils/functional.py)、[词表修复及消融记录](../../docs/PROJECT_HANDOFF.md)。

## 第二阶段序列搜索

原版 decoder 搜索将当前前缀与候选 token 组合，计算第二层输入的子空间距离，再用 `l2_span_thresh` 判断是否接受。实际代码会检查整个前缀，处理 BOS/PAD 特例，并把无法继续扩展的有效前缀作为完成序列。

原版也有启发式：保留若干未通过阈值的近似候选，用 `distinct_thresh` 抑制相似候选，支持 best-effort 输出。因此，把它描述为“纯贪心，每步只留一个 token”并不准确。

本项目使用累计分数：

```text
S(s₁…sₜ) = S(s₁…sₜ₋₁) + d₂(s₁…sₜ)
```

每次只给新增私有位置的第二层特征打分，所有扩展按累计分数排序，留下 `beam_width=16` 个。具体差异如下：

| 行为 | 原版主要路径 | 本项目 |
|---|---|---|
| 第二层接受条件 | `l2_span_thresh` 阈值检查 | 没有第二层接受阈值 |
| 前缀保留 | 通过阈值的候选及近似候选机制 | 按累计距离保留固定宽度 beam |
| 停止条件 | 无法继续扩展等条件 | 填完公开长度确定的全部私有槽 |
| 计分位置 | 原版前缀检查，含 BOS/PAD 特例 | 仅私有内容位置新增距离 |
| 已知模板和结构 token | 原版相应序列处理 | 不额外累计固定模板、EOS/PAD 位置的检测分数 |
| 同分处理 | 原版候选及去重规则 | 按 token 序列排序，保持确定性 |

这会产生两个直接后果：真实前缀可能因 beam 截断提前丢失；即使所有候选距离都很差，只要还有候选和预算，也可能填满长度并返回 `completed`。该状态只代表完成候选提交，不表示通过原版的逐位置接受条件，更不表示等于真值。

第一层距离目前只负责决定候选是否进入集合，不会再加入第二阶段累计分数。因此，第一阶段排名靠前的 token 进入集合后没有额外分数优势。

来源：[原版 filter_decoder 与 filter_decoder_step](../../dager-gradient-inversion/utils/filtering_decoder.py)、[本项目 DAGERSearch.search](dager_adapted.py)、[expansion_batch 与 retain_best](dager_search.py)。

## 数值分解与阈值

| 数值环节 | 原版主要路径 | 本项目 |
|---|---|---|
| 秩估计 | 检查最多前十个选定层，取最大数值秩 | 每层的投影后联合矩阵独立估计 |
| 分解 | `torch.svd_lowrank`，`niter=10` | `torch.linalg.svd`，`full_matrices=False` |
| 秩截断 | `rank_tol`，并限制到 `d-rank_cutoff` | 保留大于 `max(atol, rtol × 最大奇异值)` 的奇异值 |
| 默认秩参数 | `rank_cutoff=20`，容差依实验调整 | `rank_rtol=1e-5`、`rank_atol=1e-8` |
| 距离 | 归一化后 L2，另支持 L1 | 投影后归一化 L2 |
| 检测阈值 | 普通默认第一层 `1e-5`，第二层 `1e-3` | 第一层 `0.05`，无第二层阈值 |
| 子空间饱和 | 截去部分方向以保留区分能力 | 记录 `saturated`，不自动做同类截秩 |

原版 LoRA 脚本本身就把两层阈值设为 `0.05`，所以不能简单声称项目把原版阈值从 `1e-5` 放宽到了 `0.05`，应比较具体实验路径。

本项目默认 `analysis_dtype=float64` 只作用于子空间分析。受害模型、上传梯度和候选前向仍遵循声明的精度；把已经舍入的梯度转换成 float64 不会恢复丢失的信息。原版也有不同计算精度的实验分支，不能把原版统一视为无限精度实现。

当本项目得到零数值秩时返回 `no_signal`；没有 token 通过筛选时返回 `no_candidates`。子空间饱和只被记录，不会触发原版那样的强制最大秩截断。

来源：[原版分解与距离](../../dager-gradient-inversion/utils/functional.py)、[原版 get_matrices_expansions](../../dager-gradient-inversion/utils/models.py)、[原版参数](../../dager-gradient-inversion/args_factory.py)、[本项目子空间代码](dager_subspace.py)、[本项目默认配置](../../configs/attack/dager_adapted.yaml)。

## 部分前向与可选梯度重排

两者都利用因果 attention：验证前缀时，主要只需执行第一个 Transformer block，获得第二层 attention 的输入，不需要每个候选都做完整反向传播。

本项目通过 `LlavaTextView` 重建与受害者相同的图像特征、模板、位置编号和因果 mask，不把尚未填入的未来私有槽送入前向。当前没有跨前缀复用 KV cache，每批扩展重新运行相关前缀。

若设置 `rerank_candidates>0`，项目对最终 beam 中排名靠前的若干完整候选重新计算上传更新，按以下相对 L2 误差选择结果：

```text
score = Σθ ‖ĝθ - gθ‖₂² / max(Σθ ‖gθ‖₂², 1e-20)
θ 遍历实际上传参数集合
```

这里使用实际上传的完整参数集合，可能包括其他 LoRA 参数及 F-CL 的 connector，范围大于前两层 Q/K/V-A。这是明确的额外适配，原版标准搜索不包含该重排步骤。

默认 `rerank_candidates=0`，已有完整实测没有使用这一步。重排只能在保留下来的完整候选中选优，不能找回在 token 筛选或 beam 截断中已经丢掉的正确答案。

两者标准 DAGER 路径都不依赖外部语言模型概率先验。原仓库 README 的旧标题包含 “Language Model Priors”，但不能据此认定实际 DAGER 搜索使用了 perplexity prior。

来源：[LlavaTextView.layer_input 与 prefix_features](../../core/adapters/llava.py)、[本项目 rerank](dager_adapted.py)、[matching_loss](../objectives.py)、[原版部分前向](../../dager-gradient-inversion/utils/partial_models.py)、[原版主攻击流程](../../dager-gradient-inversion/attack.py)。

## 尚未迁移的原版功能

| 原版功能 | 原版实现要点 | 本项目当前状态 |
|---|---|---|
| 批量文本恢复 | 组织多个序列、候选去重及近似补全 | 强制 `sample_count=1` |
| BERT encoder 恢复 | EOS 推断长度、组合枚举、已恢复 token 排除、组合预算 | 只有因果 decoder 搜索 |
| FedAvg | 本地 SGD 多步权重差构造平均更新，调整秩与阈值 | 直接拒绝 FedAvg |
| 噪声梯度 | 多层距离、距离 logit 变换、异常值筛选 | 要求无防御的原始梯度 |
| 多架构及精度实验 | GPT-2、BERT、多个 LLaMA 型号及量化路径 | 限定单设备 LLaVA/LLaMA |

原版 FedAvg 扩展依赖局部训练期间隐藏表示变化较小等近似，不能理解为任意多步训练都满足原始精确恢复定理。

`scripts/dager_dp.sh` 使用 `attack_new.py`；该入口在 `get_matrices_expansions` 调用中有固定 `B=100` 的设置，不应与自动估秩的 `attack.py` 视为完全等价。原版的噪声实验支持也不能自动等同于某个经过隐私会计证明的 DP 保证。

本项目还不接受 F-C、F-2stage、未知图像、未知长度或带防御的上传。即使某些条件理论上可以扩展，当前支持检查仍然会拒绝，也没有自动回退到 TAG 等方法。

来源：[原版 FedAvg 更新](../../dager-gradient-inversion/utils/models.py)、[原版 encoder 搜索](../../dager-gradient-inversion/utils/filtering_encoder.py)、[原版噪声距离聚合](../../dager-gradient-inversion/utils/functional.py)、[attack_new.py](../../dager-gradient-inversion/attack_new.py)、[本项目支持边界](dager_adapted.py)。

## 攻击评估隔离与工程行为

原版 `reconstruct()` 同时包含真实数据的梯度生成、真值诊断、候选搜索和评估对齐。其 decoder 评估按每条参考文本选择 token 重合最多的预测；encoder 使用基于 ROUGE 的 assignment。这不代表核心前缀搜索直接使用真值，但它与本项目的提交边界不同。

本项目攻击只接收 `Observation`，先提交重建与候选分数，再由 evaluation 读取私有参考。最终选优只能依靠子空间分数或可观测更新误差，不能使用参考文本指标。公开字段及长度必须由知识条件声明，不能通过攻击配置自行把私有输入变为公开。

项目分别报告候选 token 集合的 precision/recall、排除歧义 token 后的检测指标，以及问题与答案各自的文本恢复指标。token 集合指标按唯一 token ID 计算，不反映顺序或重复次数，不能等同于句子恢复率。

精确 top-k 消融的评估名单也不能与攻击中的 `max_candidates` 混为一谈：前者按保存的分数进行等规模比较；后者是非歧义候选上限，攻击还会额外加入歧义 ID。

工程方面，本项目增加以下机制：

- JSON/safetensors 检查点，独立版本的攻击状态 schema。
- 输入、配置与源码指纹，以及 checkpoint 张量完整性检查。
- 保存词表扫描位置、beam、扩展位置和重排位置以支持恢复。
- 对时间与评估次数显式计数，预算耗尽返回 `budget_exhausted`。
- 异常保存失败状态并重新抛出，不自动重试 OOM 或改变实验条件。

一次前缀扩展或一次完整梯度 replay 计为一次 evaluation，与 batch 大小无关；词表扫描量另行计数。`iterations`、优化器学习率等连续优化参数不驱动 DAGER 搜索。`checkpoint_interval` 对前缀前向 batch 计数，词表块和阶段转换另行保存。

`init_source=random` 是通用配置中的中性默认标签；本方法没有随机候选初始化，不使用连续图像或 soft-token 优化，且拒绝传入候选初始化。`text_method=none`、`restarts=1` 由配置校验约束。

来源：[原版 reconstruct](../../dager-gradient-inversion/attack.py)、[本项目 DAGERSearch](dager_adapted.py)、[token 评估](../../evaluation/token_recovery.py)、[等规模筛选评估](../../evaluation/dager_ablation.py)、[协议](../../docs/protocol.md)。

## 已有实验与可支持的结论

截至核对日期，本项目已有以下单样本开发证据：

| 实验条件 | token 集合结果 | 完整序列结果 |
|---|---|---|
| 图像与问题公开，默认阈值 | 私有答案 token recall 为 0.5 | 答案恢复失败 |
| 仅图像与长度公开，默认阈值 | 问题与答案并集 recall 为 1/14 | 问题、答案均失败 |
| 同一样本，raw 精确 top-50 | 覆盖 7/14 个私有 token ID | 此消融未验证完整搜索 |
| 同一样本，public-residual 精确 top-50 | 覆盖 13/14 个私有 token ID | 此消融未验证完整搜索 |

完整诊断使用训练后的 F-L round 10、FedSGD、LoRA rank 8。未知问题条件下，默认过滤器返回 1 个非歧义候选和 9 个歧义候选，搜索填入 14 个私有内容位置，完成 2,030 次前缀检查和 128 个前向 batch，没有完整梯度重排。`completed` 与最终文本恢复失败可以同时成立。

由实现可以明确推断：默认阈值下候选集合已经漏掉真实 token，后续搜索无论增加多少 beam，都无法恢复完整原文。top-50 结果说明排名中存在更多信号，但没有证明第二层排序能够恢复正确顺序。此处只能确定候选遗漏构成完整恢复的障碍，不能据此断定其他所有改动的相对贡献。

后续随机、错误文本和词频对照显示，排名包含样本相关信号，但常见模板 token 也贡献不少命中。13/14 不能直接理解为 13 个样本特有信息被恢复，更不能理解为完整句子已恢复。这些实验都是同一样本的开发证据，不构成跨样本泛化结论。

替代图像实验只测试第一阶段分数对视觉子空间变化的敏感性，尚未形成可用的图像未知攻击。使用真实图像混合构造的条件属于 oracle 敏感性诊断，不能重新标记为图像私有条件。

来源：[PROJECT_HANDOFF 中的真实诊断与后续消融](../../docs/PROJECT_HANDOFF.md)、[验证范围](../../docs/validation.md)。

## 单独 Q K V 与联合 QKV 的配对实测

2026-10-02 应用户要求，分别运行 `projections=q/k/v/qkv` 四组完整攻击。复用前述同一份公开观测，不重新采集梯度，不修改受害者或公开图像。条件为 LLaVA、训练后的 F-L round 10、LoRA rank 8、bf16、FedSGD、图像与长度公开、问题和答案内容未知。

除所选投影外，全部攻击设置一致：`public_residual`、阈值 0.05、beam 宽度 16、非歧义候选上限 256、最多 10,000 次评估、3,600 秒、无梯度重排。四组都提交结果后才统一评估，没有根据参考文本重新调参。

| 投影 | 第一层与第二层秩 | Top-50 命中 | Top-50 精确率 | 问题召回率 | 答案召回率 | AP |
|---|---|---|---|---|---|---|
| Q | 8 / 8 | 7/14 | 14% | 40% | 75% | 0.1209 |
| K | 8 / 8 | 9/14 | 18% | 60% | 75% | 0.2088 |
| V | 8 / 8 | 8/14 | 16% | 50% | 75% | 0.2051 |
| QKV | 24 / 24 | 13/14 | 26% | 90% | 100% | 0.3355 |

表中问题和答案召回率均按 top-50 计算，参考集合分别有 10 和 4 个唯一 token ID。AP 为完整词表排名的 average precision。四组 top-50 都包含相同的 9 个歧义 ID，它们在该样本中都不属于私有参考 token。

差异不只存在于 top-50 一个截点：

| 投影 | Top-20 命中 | Top-50 命中 | Top-100 命中 | 最差真实 token 排名 |
|---|---|---|---|---|
| Q | 4/14 | 7/14 | 7/14 | 964 |
| K | 8/14 | 9/14 | 9/14 | 2486 |
| V | 7/14 | 8/14 | 11/14 | 813 |
| QKV | 9/14 | 13/14 | 14/14 | 98 |

K 在 top-50 优于 V，但 V 在 top-100 优于 K，不能给单投影建立不依赖候选数量的统一优劣次序。

实际完整搜索仍使用固定阈值 0.05，而不是上表的 top-k 名单：

| 投影 | 非歧义候选 | 歧义候选 | 实际候选集命中 | 问题与答案重建 |
|---|---|---|---|---|
| Q | 1 | 9 | 1/14 | 均失败 |
| K | 0 | 9 | 0/14 | 均失败 |
| V | 0 | 9 | 0/14 | 均失败 |
| QKV | 1 | 9 | 1/14 | 均失败 |

四组问题与答案的 EM、ROUGE 和 word recall 都为零。Q、K、V 生成的序列与 QKV 不同，但都没有成功恢复。K/V 可以用歧义候选填满长度，因此依然返回 `completed`，该状态不能当作恢复成功。

Q/QKV 各使用 2,030 次前缀评估，K/V 各使用 1,818 次；模型加载之后每组约 6.8 至 7.3 秒，无预算耗尽、OOM 或参数回退。GPU 5 上串行执行完成，任务及 watcher 均退出，实验显存已经释放。

重跑 QKV 的完整词表分数与之前有效词表 public-residual 消融的分数逐元素完全一致，最大绝对差为零，歧义标记也完全一致。因此本次观察到的差异不是旧 QKV 基线发生变化造成的。

这次实测支持：在这个低秩 LoRA 样本上，Q、K、V 单独使用的排名不等价，联合 QKV 的 token 排名优于三个单投影。但这是 `n=1` 的配对证据，不推翻原论文在相应满秩条件下任选一个投影的结论，也不能证明 QKV 普遍更好。完整序列恢复仍然失败；本次没有额外运行 top-k 版本的完整搜索。

复现入口：[evaluation/dager_projection_ablation.py](../../evaluation/dager_projection_ablation.py)。结果目录为 `outputs/diagnostics/dager_projection_qkv_gpu5_n1/`，包含每组检查点、候选分数、重建结果与评估，以及以下汇总：

- [summary.json](../../outputs/diagnostics/dager_projection_qkv_gpu5_n1/summary.json)：秩、候选集、排名及文本指标。
- [top50_comparison.json](../../outputs/diagnostics/dager_projection_qkv_gpu5_n1/top50_comparison.json)：等规模 top-50 结果和候选重合度。
- [ranking_validation.json](../../outputs/diagnostics/dager_projection_qkv_gpu5_n1/ranking_validation.json)：排名曲线与均匀随机对照；不替代词频和错误文本对照。
- [plan.json](../../outputs/diagnostics/dager_projection_qkv_gpu5_n1/plan.json)：运行前固定的设置与输入哈希。

## 验证记录与文档时效

2026-10-02 的本次实现对比期间执行了：

```bash
/tmp/giavlm-venv/bin/python -m pytest -q tests/test_dager.py
```

结果为 `24 passed in 4.89s`。测试覆盖数学恒等式、秩不足反例、部分前向一致性、合成恢复、预算与断点恢复等。

这些测试使用合成矩阵和随机小型模型；它们证明相关实现和接口在测试条件下成立，不替代真实预训练模型上的恢复实验。最初的实现对比没有重新执行 GPU 实验；随后按用户要求执行了上一节记录的四组配对 GPU 实验，并再次通过 24 项 DAGER 专项测试和新实验入口的 Ruff 检查。

核对时 [`docs/baselines.md`](../../docs/baselines.md) 仍有“真实 token 检测与重建尚未评估”的旧句子，已落后于当前交接文档。方法实现与适用范围应结合代码核对，实验状态应以较新的交接和验证记录为准。
