# 项目交接

## 2026-11-05 医学合成原型续测完成

- 用户明确授权以文生图补足并完成第三项。全新三十二张 Medical synthetic 对照等数量 Generic photo，预定 Q24 在两模型、SLAKE 与 VQA-RAD Natural QA 均为 20/20；结果与图已并入 [[gradient-prototype-transfer]]，方案更新在 [同一方法计划](plans/2026-11-05-gradient-prototype-method.md)。原纯真实公开库停止记录不改，不提交 Git，保留其他会话工作。
- 当前优先级是冻结此条件、在新目标确认；本轮仍是已分析的四十图，新增公开库而非独立医学队列。Qwen Q32 Natural 仅14/20、15/20，CT小组较弱；不要据当前目标继续调提示、选层或阈值。所有公开阈值按原准则都为0，收益来自原型替换，不是目标校准。
- Q24 在 SLAKE CT、MRI 与 VQA-RAD CT 内均8/8、两模型一致；只能支持部位信息有贡献，不能把结果外推为任意属性、领域未知或临床有效性。CT/MRI仅提示分组，若干胸部图偏X-Ray、一张为心脏渲染，全图按部位规则接受，无换图。两库负类覆盖、语义和有效家族数不同，尚未隔离纯医学风格贡献。
- 输出 outputs/diagnostics/prototype_medical_synthetic_20261105，base2fd25b6；source-medical-generation-v1、source-medical-probe-v1、source-medical-analysis-v1 分别186、187、187文件。原gia、gia-qwen35、qwen-image-21环境和模型未改，只加evaluation、测试、文档，生成图与逐图预测仅在outputs。
- 本机GPU5、UUID GPU-8ad9e9b4-e159-4f78-7036-2d1d018e7b7d，生成前0MiB；probe前余57734MiB，按共享授权继续同卡顺序。生成controller2612178/worker2612180/watcher2612179、probe controller2630316/watcher2630317、GPU workers2630320/2635954、CPU workers2641595/2645174均已实际核验退出，全部returncode0。最终卡仍有他人23403MiB占用，本轮已释放，不宣称整卡空闲。
- 生成32图405.54秒、controller462.33秒；LLaVA196.61秒/峰值21099.19MiB，Qwen183.67秒/12504.18MiB，probe及评分controller638.88秒。原生门槛448/496张量逐值一致，模型指纹和TrainingSpec与原上传一致，两模型全部评分提交后才评价，无OOM、自动重试或条件变更。
- 独立重建24组原型、2688组直接SVD距离及960组手工指标通过，误差0；全部公开阈值复现，32生成像素不同，与149张排除目标无像素重叠。无待运行实验，主报告完整，后续需新目标确认。
- 完整CPU回归520通过、4跳过，九项定向、Ruff、21份记录与产物核验、MkDocs严格构建通过；三份源码快照、四个新发布文件及前十六个发布文件哈希一致，提交后的原型、预测和规则不变。finalized.json及phase-status.json记录完成，无待运行任务或Git提交。

## 2026-11-05 原型边界偏移查证完成

- 用户授权执行并允许本机、WN-A100 优先空闲卡，否则共享最低显存占用和利用率卡；已扫描本机 GPU5 空闲、WN-A100 全忙，但新医学原型对照在数据门槛停止，因此没有启动或占用 GPU。前两项仅 CPU 已完成，结果与两张聚合图追加 [[gradient-prototype-transfer]]，方案状态更新到 [同一方法计划](plans/2026-11-05-gradient-prototype-method.md)。不提交 Git，保留其他会话修改。
- 当前结论：统一 Fixed QA 后仍有域间边界偏移，问答不同不是唯一解释；预定相对能量归一化也没有让公开边界迁移。不要在既有四十图上继续调阈值、翻分数或选层；若继续纯真实辅助库条件，需补足至少两张明确主要显示头部、与全部目标隔离的公开 CT，重新声明来源和冻结划分。
- 训练辅助池十五张元数据头部 CT 全部目视核验，仅六张明确符合主要头部定义，另九张颈部主导或边界不清，少于所需八张。按原停止规则保留 visual-review、medical-bank-stopped，未用目标图补齐或降低平衡要求；不能据此评价医学原型方法有效或无效。原图与拼图只在 outputs。
- 根目录 outputs/diagnostics/prototype_boundary_20261105，base2fd25b6；source-boundary-v1、source-boundary-report-v1 冻结执行和独立分析，/tmp/giavlm-venv/bin/python，CUDA_VISIBLE_DEVICES 为空。CPU 主计算 197.88 秒，controller 墙钟 200.67 秒；controller2580229、worker2580233、watcher2580231，returncode0，报告也 exit0，无待运行实验。初次沙箱后台封装未开始 worker 就退出，原 launch 与说明保留，没有评分重试或条件变化。
- 原 1536 组距离重放最大误差 3.28e-15，96 组独立 SVD 几何、288 组手工分类及全部公开阈值复核通过；新分数与公开阈值全部提交后才加入医学标签。核心训练、攻击、模型和旧分数不变；新增代码只在 evaluation 和 tests。
- 完整 CPU 回归 519 通过、4 跳过，十三项定向、Ruff、21 份记录与产物、MkDocs 严格构建通过。执行及分析快照 184、185 文件，六个新发布文件与前十个发布文件哈希一致，当前代码与冻结源码一致；finalized.json 记录两项完成、医学原型对照因数据不足停止，无待运行实验。

## 2026-11-05 E1 新增命中审计完成

- 用户要求核查 E1 是否主要补通用问题 token。复核原已保存候选，Noise Random 的 Blind 相对 B anchor 新增 136 次命中，涉及 11 个 ID，98.5% 为问句结构、图像通用词或标点；相对 Residual 的解剖新增来自 B 通道。Top-k 仍会丢医学内容词片，完整覆盖没有增加；结果见 [[dager-upload-defense]] 的 E1 审计。
- 新模块 evaluation/question_filter_token_audit.py，仅 CPU 事后评价，不跑 GPU、模型前向或新预测；144 组统计、6694 次事件、27/27 分词对齐及通道归属独立复核通过。聚合在 docs/experiments/data/federated-text-pilot/e1-token-gain-*，逐 token 私有明细仅在 outputs/diagnostics/question_filter_defense_20261105/token-gain-audit/。
- 原评分与候选未改，原清单未改，无待运行任务，不提交 Git；保留其他会话和 paper 修改。29 项定向与索引测试、Ruff、21 份记录及产物检查、MkDocs 严格构建、原 23 个哈希与 diff 空白检查通过。

## 2026-11-05 问题筛选六方向验证完成

- 用户授权的六个Stage-1方向全部完成，方案 [DAGER 问题筛选验证](plans/2026-11-05-question-token-defense-validation.md)，结果与三张聚合图见 [[dager-upload-defense]]。无答案线索的问题先验改善部分覆盖，答案条件化未形成稳定额外收益；L1 up、L2 K仅有局部首词线索，不能称完整问题恢复。
- 所有完整筛选主方案未通过推进门槛；next-phases.json明确P2组合、P3新样本、P4序列不运行。没有新ROUGE、Word Recall或Joint EM，没有待运行任务，不重复启动。下一步若继续，优先冻结两个首词位置，用公开数据改进提议池后在新目标确认，不能在旧三十条继续选层或调提示。
- 根目录 outputs/diagnostics/question_filter_defense_20261105，base2fd25b6；source-question-v1 SHA256 68898372ed8d30275e6f52aae19d28c67e9f3d91e3d21e7819ae7ab2df908ada，四个评分模块未变，gia和模型不变。新文件 evaluation/question_filter_*、tests/test_question_filter.py 均未提交，保留其他会话与paper修改。
- WN-A100只用GPU2，UUID GPU-8cb10863-d4c8-528f-3add-2632ae9cb39d；controller2518、workers2525/2591/2629/2738/2768/2782、watcher2530经SSH核验全部退出，六阶段returncode=0，总墙钟1311.05秒。最终显存4MiB、利用率0；没有占第二张卡或自动重试、改变条件。
- 原生448张量逐值一致、180份原上传完整哈希、10380组预算、811个提交文件、72项历史指标、83040项独立指标及2768项聚合复核通过。53项定向与索引测试、Ruff、21份记录及本地产物、MkDocs严格构建、23个发布/代码哈希和完整冻结源码指纹均通过；finalized.json记录完成，没有待运行任务或Git提交。发布聚合位于 docs/experiments/data/federated-text-pilot/question-filter-*，图在 figures/federated-text-pilot/question-filter-*，不含私有图文。

## 2026-11-05 问题筛选实验方案

- 六方向方案已执行，以上方完成记录及 [[dager-upload-defense]] 为准，不再等待执行确认。设计文件保留具体对照、指标、门槛与停止条件，未达到门槛的后续阶段没有启动。

## 2026-11-05 梯度原型方法三项验证完成

- 用户授权的方法问题、理论边界和三项冻结方案集中在 [梯度原型方法计划](plans/2026-11-05-gradient-prototype-method.md)，结果与两张聚合图已追加 [[gradient-prototype-transfer]]。候选主方案未解决医学迁移；不要重复启动、按旧目标选择赢家或补调阈值。
- 本轮仅 CPU，复用两模型的既有公开特征和一百六十份医学上传，没有新图或新捕获；医学四十图为重复目标诊断，旧 public dev 重分 calibration 与 audit 也不算新公开确认。六十一套真实库及二十套标签置换库全部报告，主算法用唯一满库。
- 下一优先级是解释公开到医学的分差偏移，以及公开模块指标大量并列的选择问题；是否加入公开医学辅助数据须单列领域先验，再冻结规则、用新目标确认。理论上的输入行空间包含关系不是属性恢复保证，当前扰动界过松。
- 根目录 outputs/diagnostics/prototype_method_20261105，base2fd25b6；执行与报告快照 source-prototype-method-v1、source-prototype-method-report-v1 分别 174、175 个文件，全部哈希通过。/tmp/giavlm-venv/bin/python，CUDA_VISIBLE_DEVICES 为空；主计算 184.79 秒，主运行和报告均 exit 0，无待运行作业。
- 独立复核 1536 组满库距离、480 组直接距离、48 组主指标、5832 行抽样指标通过，公开规则可复现、分数提交顺序通过、校准不改变同模块排序；全部已提交分数与六个发布文件哈希不变。新代码只加 evaluation 和测试，不改核心训练或攻击，不提交 Git，保留所有其他会话文本方向与 paper 修改。
- 完整 CPU 回归 501 通过、4 跳过，Ruff、21 份记录与本地产物核验、MkDocs 严格构建通过；finalized.json 与 phase-status.json 记录完成。本轮无待运行实验。

## 2026-11-05 问题 token 改进思路

- 用户要求核查外部五个建议、结合已有思路并按成功可能性整理；idea 文档为 [DAGER 问题 token 改进](plans/2026-11-05-question-token-defense-ideas.md)。本轮只写文档，没有启动 GPU、修改攻击代码或既有评分，不提交 Git。
- 按固定预算下新增问题候选证据的相对前景，依次为答案条件化问题先验、Top-k 支持集正则拟合、首层 K/V 输入能量、实际前缀浅层 A、配额与 B 共识、修正后的中层 B 探针。排序为主观研究优先级，不是测得概率；K/V 可因成本低先作排除诊断，不自动执行。
- 已纠正浅层 MLP 可用单词 embedding 精确预计算的错误、输出词方向绝不可能响应问题的过强断言、将 Post-mask 当 Masked ridge 的混淆，以及短答案配额、跨位置独立降噪和完整恢复的未经验证保证。支持集公开基可能覆盖全部观测坐标，必须防止拟合掉私有信号。
- 后续应补充问题独有 token 的召回与问题完整覆盖，先保存候选后评价；先单机制、再等预算融合与新样本确认，有候选收益才接序列搜索。实验结果仍扩展 [[dager-upload-defense]]；本轮没有新实验结果或待运行任务。
- 四行基线与 L31 down 指标已从原聚合核对；术语定义同步 glossary 和 abbreviations。24 项索引测试、21 份记录及本地产物、MkDocs 严格构建和 diff 空白检查通过；没有新实验产物或 Git 提交。

## 2026-11-05 整体候选指标展示核对

- 用户询问新的 B 读出是否有整体 Recall、ROUGE。既有 last-b-* 聚合都包含 question_recall、answer_recall、joint_recall，正文之前偏重答案；已在 [[dager-upload-defense]] 的全层扫描下补充两种图像的整体候选召回表，并解释按 Q∪A 去重 token 集合逐样本计算。
- 新 B 方案只测试 token filter，尚无新 Word Recall、ROUGE 或 Joint EM；旧序列结果不能移作新方案结果。下一阶段若按原 PPT 指标比较，必须把固定候选方案接入序列搜索后再评价，不能将 Answer recall 当作完整文本恢复率。
- 只核对既有聚合，无 GPU、攻击或新参考文本读取；更正 None 条件 L32 o、L31 down 的 GT Joint FullCoverage 两处抄录为 27/30，保留 Random 26/30。没有修改任何分数、候选或聚合产物，不提交 Git。
- 索引 24 项、21 份记录及本地产物、MkDocs 严格构建通过，原扫描 23 个发布/代码 SHA256 不变；没有待运行任务。

## 2026-11-04 LoRA-B 全层扫描完成

- 用户确认扫描所有层；计划 docs/plans/2026-11-04-last-b-layer-sweep.md。32 层 down_proj.B 与 o_proj.B 共 64 位置，复用 SLAKE30 的 None、Noise、Top-k 上传及原 LM prior，固定同一能量评分和 80/20 补充，每组仍 109 候选；不跑序列搜索、不合并位置、不提交 Git。
- 新入口 evaluation/last_b_layer_sweep.py、evaluation/last_b_sweep_math.py；缓存公开加权词向量和排序，不改变评分定义。六项新数学/边界及既有位置读出共 20 项测试、Ruff 通过；三个历史位置逐分数、候选与指标复现是硬门槛。
- 输出 outputs/diagnostics/last_b_layer_sweep_20261104/，source-sweep-v1 SHA256 3222549a7254438dc0198804b70f2f6d3867ac714d5ffc265994ba2e53020f83、base2fd25b6、gia 不变。启动前复查 WN-A100 GPU 0 空闲，UUID GPU-25dab3f9-1eda-bcfd-291e-9ca1ca3b673b；controller 2386、GPU worker 2391、watcher 2396，单卡 detached，没有占用第二张卡。
- 全部 64 位置完成，两个模块的开发候选按预设跨防御均值仍为 L31 down、L32 o；高值集中于后部相邻层，Top-k 单项召回是 L29 o 略高，但答案完整覆盖低于 L31 down。所有位置的 Random Joint FullCoverage 最大仍为 Noise 0/30、Top-k 1/30；结果在 [[dager-upload-defense]] 的 B 全层扫描，不继续同批扫参数或多层融合。
- 所有分数及固定候选先保存后评价。三个历史位置的 270 份分数差零，2160 组历史候选及指标复现；271 个评分/候选哈希、全部预算 109、768 次原实现候选重算、聚合/配对/排序独立审计通过。
- WN-A100 controller 2386、GPU worker 2391、CPU 评价 2487、watcher 2396 经 SSH 核验全部退出，两个阶段 returncode=0；墙钟 179.26 秒、峰值 allocated 16.65 GiB，最终核验该卡显存零、利用率零。没有待运行任务，不重复启动。
- 聚合表图在 docs/experiments/data/federated-text-pilot/last-b-sweep-* 与 figures/federated-text-pilot/last-b-sweep-*，不含私有图文或原始梯度。下一步只应以冻结的两个模块候选在新样本确认，不把本批开发排序称作泛化最佳；不提交 Git。
- 最终 44 项定向及索引测试、Ruff、21 份记录及本地产物、MkDocs 严格构建、23 个发布/代码 SHA256 与当前/冻结实验源码一致性通过。评分、评价、controller 和 watcher 均结束，保留其他会话修改，没有 Git 提交。
- 开发主排序固定为 Random 的 Noise/Top-k 补充答案 recall 均值，按预设规则打破平局，每个模块保留一个确认候选。只报告同批探索，独立新样本确认列为后续阶段，不把开发择优称作泛化最优。

## 2026-11-04 LoRA-B 三位置对照完成

- 用户授权验证末层 down_proj、末层 o_proj、倒数第二层 down_proj；计划 docs/plans/2026-11-04-last-b-position-study.md，独立入口 evaluation/last_b_positions.py。复用同一 SLAKE30 的 None、Noise、Top-k 上传、同一最终词方向及能量评分；各位置分别比较，不相互融合，不跑序列搜索、不提交 Git。
- 原 Residual、公开 LM prior、循环错配对照及固定 80/20 补充保持一致；全部三个位置分数与候选保存后才评价。末层 down_proj 必须逐分数与候选复现上轮，避免把实现差异误当位置收益。新位置与上轮均是同批机制诊断，不称独立验证。
- 五项位置与边界测试，连同上轮读出共 14 项通过，Ruff 通过；输出 outputs/diagnostics/last_b_positions_20261104/，source-positions-v1 SHA256 775c0093ddcc2878ff72b92e85dcb078eecfbd1fcfe2fcf550343f27674d0f09、base2fd25b6。既有公开 LM prior 原样复用，不再生成。
- 两个新位置的答案候选优于原 L32 down，L31 down 的固定补充均值通常较高，L32 o 的单独 AP 更高；没有一个位置全面最优。完整问答候选覆盖仍只有 Noise 0/30、Top-k 1/30，未跑序列搜索，不自动合并位置或扩展层数；完整结果见 [[dager-upload-defense]] 的 B 读出位置对照。
- WN-A100 GPU 1、UUID GPU-6a309ac1-08fa-062a-b449-e8de39ba4d09，启动前空闲；controller 2254、GPU worker 2259、CPU 评价 2353、watcher 2264 均经 SSH 核验已退出，两阶段 returncode=0。墙钟 134.68 秒、峰值 allocated 14.42 GiB，最终核验该卡显存零、利用率零；没有待运行任务。
- 原 L32 down 全词表分数差为零，1080 组历史候选及原指标复现；全部 2160 组候选均为 109，271 个评分/候选哈希、聚合与配对变化重算通过。聚合表图位于 docs/experiments/data/federated-text-pilot/last-b-positions-*、figures/federated-text-pilot/last-b-positions.* 与 last-b-supplements.*，无私有图文；不提交 Git。
- 最终 38 项定向与索引测试、Ruff、21 份记录及本地产物、MkDocs 严格构建、16 个发布/代码 SHA256 及当前/冻结实验源码一致性通过；保留其他会话修改，没有待运行的本轮任务。

## 2026-11-04 末层 B 答案读出完成

- 通过预设答案信号门槛，固定 80/20 融合改善防御下答案覆盖，优于同配额 LM 补充。问题侧仍漏检严重，完整问答候选覆盖只有 Noise 0/30、Top-k 1/30，且 LM 补充也达到相同完整覆盖；未运行序列搜索。这是同批机制诊断，完整结果与对照见 [[dager-upload-defense]] 的末层 B 答案读出。
- 输出 outputs/diagnostics/last_b_token_readout_20261104_v2/；固定末层 down_proj B，全部主候选先保存后评价，未运行序列搜索，不提交 Git。
- WN-A100 GPU 1、UUID GPU-6a309ac1-08fa-062a-b449-e8de39ba4d09，GPU worker 1935 已结束；原 controller.json 保留检查器误设词表的报错，completion-controller.json 记录本机 CPU 续评终态。原父进程未收集 worker 退出码，审计保留 null；全部评分及候选已独立核验，不重跑 GPU。不要把本批机制诊断称作独立验证。
- GPU worker 到全部提交为 122.92 秒、峰值 allocated 14.42 GiB；本轮全部远端进程已退出，该卡随后被其他任务使用，不宣称整卡空闲。本机没有后台续办任务。最终 48 项定向与 24 项文档索引测试、Ruff、21 份记录及本地产物、MkDocs 严格构建、180 组原候选、451 个主评分/候选哈希与 18 个发布/代码 SHA256 通过；没有 Git 提交。

## 2026-11-04 图像属性跨模型测试已完成

- 用户授权先测试其他模型和数据集，具体冻结矩阵已追加 docs/plans/2026-10-28-image-property-next-stage.md。Q32 预先声明主条件，Q24 与 K/V 次要；不按本轮公开或私有成绩重新选择。SLAKE A 二十图重复配对，VQA-RAD 二十新图，Head 正负各十；原始自然问答和固定通用问答配对，共 160 份医学上传，另外两模型合计 96 次公开 dev 梯度回放。
- 模型使用现有 crossmodel_text_20261017/models/slake 的 LLaVA-1.5-7B 与 Qwen3.5-4B Round10、Rank32、alpha64、Q64/A64 快照，每模型同一快照评价两个数据集；不重训。新 LLaVA 权重及容量与历史属性实验不同，必须重报基线，不能直接把差值归于架构。Qwen24/32均为 full_attention；不将 DeltaNet 映射成 QKV。
- 原公开 48 fit、48 dev 非医学三风格图共用，每模型重新提取实际 LoRA-A 视觉输入原型。医学推断只读取匿名 Observation 张量，不给问答、真图、长度或标签，不重构文本。全部两模型预测提交后才统一读医学标签。VQA-RAD 先看完 57 eval 图再按固定规则抽样，CT 含正负类，MRI 只正、X-Ray 只负；排除与全部 SLAKE 图像的像素重叠，患者独立性未知。SLAKE 剩余未用 eval 头部不足，不冒称 A 为新确认。
- 入口 evaluation/crossmodel_property.py 与 evaluation/crossmodel_property_report.py；最终输出 outputs/diagnostics/crossmodel_property_20261104_v3，source-crossmodel-property-v1 与 source-crossmodel-property-report-v1 各 169 文件冻结，base2fd25b6，原 gia 与 gia-qwen35 环境不变。完整预定模块名在两快照元数据核对，原问答两 tokenizer 容量审计通过，无截断。
- 首轮 WN-A100 GPU2，UUID GPU-8cb10863-d4c8-528f-3add-2632ae9cb39d，启动前 0 MiB；controller1687、watcher1688、LLaVA worker1689 已停止退出。在公开阶段发现逐上传 core.capture 全模型指纹主导时间，尚未捕获或评分医学目标；记录先写入 [[gradient-prototype-transfer]]。原目录及 abandoned 保留，不重复启动。
- 优化仅缓存不变模型指纹，仍调用原 simulate_update；需首尾完整指纹与首份原 capture 全张量逐值核对，小模型等价测试已通过。v2 只准备、从未启动；当前可运行冻结版本在 outputs/diagnostics/crossmodel_property_20261104_v3，source-crossmodel-property-v1、source-crossmodel-property-report-v1 和 plan 已冻结。样本、问答、模型、梯度和评分规则不变，私有数据未重采样；后续报告入口 evaluation/crossmodel_property_report.py，全部四格结果需要重建本轮 LLaVA 基线。
- 用户已明确回复“允许”，授权本轮共享 WN-A100 显存占用最少且剩余充足的 A100。复查选择 GPU0，UUID GPU-25dab3f9-1eda-bcfd-291e-9ca1ca3b673b，原占用 7787 MiB、剩余 73366 MiB；单卡串行两个模型，精度、任务、样本与评分不变。controller2039、watcher2040、LLaVA worker2041、Qwen worker2139 经远端核验均已退出；本轮占用释放，同卡其他任务仍有 8335 MiB，不宣称整卡归零。没有待运行阶段，不重复启动或占第二张卡。
- 总墙钟 731.73 秒，LLaVA 333.76 秒、峰值 allocated21099.19 MiB，Qwen375.74秒、12504.18MiB。原生 capture 等价门槛分别覆盖全部448、496张量，逐值一致；首尾模型指纹一致。v3无OOM、重试或协议变化。
- 全部160份医学预测提交后才统一评价，1536组距离及12个原型独立重算误差零；原eval行、四十图跨模型与QA配对、目标像素去重、VQA-RAD与全部642张SLAKE图像无像素重叠核验通过。结果、完整主次配置、问答排序与分类差异、同模态交换对照已并入 [[gradient-prototype-transfer]]；Q32不是通用读出层，Qwen24层次要信号需公开冻结规则与新目标确认，不在本批医学标签上择优或调阈值。
- 七项相关测试与 Ruff 已通过；首轮全 CPU 的唯一失败是文档索引过期，重新生成后索引及新测试28项通过，准备阶段完整CPU回归485通过、4跳过。最终文档与定向43项、Ruff、21份记录与本地产物检查、MkDocs严格构建、两个各169文件快照与四个发布SHA均通过。发布文件位于 docs/experiments/data/gradient-prototype-transfer/crossmodel-*，只有聚合表图；私有图文、逐图分数和映射不发布。
- 不另建实验页、不提交 Git，不自动追加新属性、模型或目标；保留所有其他文本防御、文档拆分与 paper 修改。

## 2026-11-04 LoRA A/B 防御筛选验证完成

- 用户授权先写计划、验证后记录结果，不提交 Git；计划在 docs/plans/2026-11-04-lora-ab-defense.md。只复用 SLAKE30 的 None、Noise 原上传，固定 GT/Random 和 109 候选，比较原 Residual、A/B 一致性校正、残差左空间长度截断、带 Gaussian null 门槛的左空间截断；不运行序列搜索。
- 结果已并入 [[dager-upload-defense]] 的 A/B耦合：三种新规则都没有改善 Noise 下的完整覆盖。A/B 一致性校正虽满足约束，却不足以修复漏检；左空间投影降低矩阵误差后仍未改善覆盖。保留原筛选器，不追加序列搜索或参数扫描。
- 独立入口 evaluation/lora_ab_filter_study.py、evaluation/lora_ab_math.py，数学及原筛选相关测试 39 项通过，Ruff 通过；不改生产攻击。全部 480 组分数保存后才评价 token，并单独标记事后配对干净上传诊断；120 组原候选精确复现、240 个张量文件哈希及聚合指标重算通过，全部候选预算 109。
- 首轮在首份评分前因 CPU 观测梯度与 CUDA 基不在同一设备而失败，未读 token 真值；失败记录并入 [[dager-upload-defense]] 的运行记录。原目录 outputs/diagnostics/lora_ab_filter_20261104/ 不覆盖。显式修正设备后另冻结 source-v2，输出 outputs/diagnostics/lora_ab_filter_20261104_v2/，SHA256 为 5ef098123fc6f87a321443471f96a684bf99aacb56b0b064d5a3fd546de7ad16，base commit 2fd25b6。
- 本机 6793b5d1b465 的 gia 环境、GPU 2（UUID GPU-ce1827e8-5b97-c17f-66da-3acf01a97692），GT/Random 串行；仅用一张本机卡，不占 WN-A100。controller 2037999、GT 2038004、Random 2042508、CPU 评价 2046195、报告 2046795、watcher 2038035、显存巡检 2038037 全部退出。修复后墙钟 5.11 分钟、峰值 allocated 15.83 GiB；最终卡上 4 MiB、利用率零，无待运行任务。
- Gaussian gate 额外假设防御比例为 0.01，属于明确标记的防御感知诊断，原 Observation 没有该参数；其余两种耦合规则不需要该比例。配对干净上传只在全部分数提交后用于误差诊断，不能用于 noisy 评分或选参。
- 十份聚合表图与审计及 manifest 位于 docs/experiments/data/federated-text-pilot/lora-ab-*、figures/federated-text-pilot/lora-ab-filter.*；不含私有图文或原始梯度。定向 39 项、文档索引 24 项、Ruff、21 份记录及本地产物、MkDocs 严格构建、16 个发布/代码 SHA256 和冻结源码一致性全部通过；全部未提交，保留其他会话修改。

## 2026-11-02 图像属性与联邦文本记录拆分

- 用户确认将 [[image-property-ideas]] 按研究问题拆分，取代 10-28 "统一为一页"的安排。原页改为五个 idea 的状态总览；详细结果在 [[gradient-prototype-transfer]]（深层梯度原型与医学迁移）、[[candidate-scoring]]、[[visual-concept-probe]]、[[convex-token-recovery]]。
- 结论修正：第 32 层 Q 在 A、B、C 三批都能判断头部（Head Balanced accuracy 0.750、0.700、0.854），A 批 CT、MRI 组内分别 6/8、8/8，不能用模态或字面词解释；[[gradient-prototype-transfer]] 的 outcome 改为 mixed。公开选定的第 24 层与 V24 仍失败。
- 当前优先：把 Q32 头部判断作为预先声明的主条件，在新的确认批次上验证（同一模态内多个部位，加固定通用问答配对上传）。
- 同日将 [[federated-text-pilot]] 拆为两页：原页改名"DAGER 联邦本地更新"，只保留多样本混合、参数精度、本地步数与 AdamW；防御、退化诊断、部分消除、谱加权与动态候选移到 [[dager-upload-defense]]。划分记号统一为 BxSy（如 B2S1），样本集记为 SLAKE30、SLAKE28、Pilot4，已加入 glossary。
- 两次拆分前的完整页面未提交 Git，备份在 outputs/diagnostics/docs_backup_20261102/。只改了文档，没有运行实验，不提交 Git。

## 2026-11-01 A 批 K/V 补测已完成

- 用户明确授权补测最早医学与通用候选梯度匹配的 A 批二十图。固定批次名及实验映射已加入 [[gradient-prototype-transfer]]：A 为最早十头六腹四胸；B 为后续多层与固定问答的五头十腹五胸；C 为首轮 K/V 新采样的八头两腹十胸。A/B/C 图像 ID 互不重叠；首轮 K/V 的 historical 是 B，不是 A，避免按相对“新旧”继续混淆。
- 新入口 evaluation/medical_kv_cohort_a.py，沿用公开冻结 V24、全部六个 Q/K/V 层模块、原非医学视觉原型和原始距离规则。复用 A 原完整上传，不恢复文本、换问答、重捕获或生成图像；实际评分只接收张量与公开原型。全部匿名预测提交后才评价标签，未按目标结果拟合或选层。
- A 原协议公开长度，学习率及 rounds、snapshots 配置元数据与 B/C 不同；分类器不读长度，模型指纹相同。当前 FedSGD 上传原始梯度，lr 仅服务端应用更新，不影响固定快照的捕获。保留原协议，不称作长度未知协议的严格重复。
- 结果见 [[gradient-prototype-transfer]] 的 A 批补测：K/V 全部判负，与 B/C 的负结果一致；Q32 头部复现既有结果。A 的 CT、MRI 小组存在排序信号，但不是已验证的类别修复，不在 A 上校准阈值或据此选择新模块。
- 输出 outputs/diagnostics/multilayer_property_20261101/medical-kv-a，source-medical-kv-a-v1 冻结，基础 commit 2fd25b6、/tmp/giavlm-venv/bin/python；仅本机 CPU 评分 14.26 秒，无 GPU、SSH 或长任务。评分与独立审计均已退出，原 Q 与 240 组独立 SVD 距离误差零，指标与配对区间重算一致，不重复执行。
- 发布 medical-kv-a-* 三份聚合、CSV、审计及 manifest，不含私有图文与逐图映射；B/C 原发布文件保持不变。运行前四项定向、最终文档及定向 47 项、Ruff、十六份记录与本地产物、MkDocs 严格构建、164 项源码与发布 SHA 全部通过。无待运行任务，不提交 Git，保留其他图像、文本防御及 paper 改动。

## 2026-11-01 K/V 医学迁移已完成

- 用户明确授权真实医学数据集迁移。入口 evaluation/medical_kv_data.py、evaluation/medical_kv_transfer.py；保留旧二十真实图的原 private 自然问答上传，另采样二十新 SLAKE eval 图。所有四十图全部盲评提交后再评价，不重构文本，不改问答，不向分类器提供真图、标签或长度，不提交 Git。
- 主条件在读取任何医学模块分数前，仅按公开六 QA 两属性最差 BA、均值、较浅层和 Q/K/V 顺序冻结为 L24_v；全部六个层模块组合仍报告。旧二十图只作重复诊断，新二十图按图像 ID 和像素排除八十三个此前属性目标，患者关系未知，不在医学结果上改选模块或阈值。
- 新队列元数据为八 MRI 头、五 CT 腹、七胸部 X-Ray；预捕获视觉核验发现三 CT 为主要胸部视野，固定原图集合更正标签，最终八头、两腹、十胸。腹部仅两正例，CT 内有两腹三胸；混合视野按主要部位定义，模态混淆仍在。所有更正与私有样本仅留在 medical-kv/private，审核后已冻结。
- 结论与完整分层结果见 [[gradient-prototype-transfer]]：公开冻结主条件及其余 K/V 对照没有成功迁移，旧、新两批的两属性均全部判负；Q32 头部对照有信号，但仍受模态与文本混淆，不能据此更换主条件。新腹部只有两正例，全负的高表面正确率不代表恢复成功。
- 根目录 outputs/diagnostics/multilayer_property_20261101/medical-kv，source-medical-kv-v1、source-medical-kv-report-v1、基础 commit 2fd25b6、现有 gia。本机 GPU 均忙，用 WN-A100 GPU 0，UUID GPU-25dab3f9-1eda-bcfd-291e-9ca1ca3b673b；启动前空闲。远端 controller 1424、worker 1426、watcher 1425 已全部退出，本轮 GPU 占用已释放；结束后该卡另有其他任务，勿停止或宣称整卡空闲。日志 local-medical-kv/controller.log，墙钟 460.46 秒、峰值 allocated 20526.11 MiB，无 OOM、重试或条件变更，不重复启动。
- 原公开原型与直接行空间距离比较沿用，旧 Q 距离重放误差零，低于预定 1e-12 容差。独立复核四十份上传、480 组模块距离、168 组指标均通过，全部八十三旧图像 ID 与像素排除通过；预测在评估标签合并前已提交。发布 medical-kv-* 聚合与图表，不含私有图文。
- 完整 CPU 回归 463 通过、4 跳过；十项定向及最终文档与定向 46 项、Ruff、十六份实验记录与本地产物检查、MkDocs 严格构建通过，两个冻结源码与发布文件 SHA 全部核验。没有待运行的本轮任务；下一步若换医学公开原型或归因领域差异，须先声明新条件，不能在本批目标上调参。保留其余图像、文本防御与 paper 修改，不提交 Git。

## 2026-11-01 问答控制与 QKV 实验已完成

- 用户授权执行上轮三个控制：同问题换长短回答、同回答换问法、比较第 24/32 层 Q/K/V；另按计划在六个公开内容家族做四组问答的逐 token 损失分解。只用既有三十三张公开审核图，十二 QA 条件，共 396 次原生完整回放，不读取医学目标，不改模型或攻击核心，不提交 Git。
- 入口 evaluation/qa_type_controls.py，冻结细案已在唯一下一阶段计划追加；结果继续汇入 [[gradient-prototype-transfer]]。根目录 outputs/diagnostics/multilayer_property_20261101/qa-controls；主配对为同确认问题的三 token Yes. 与八 token 等义长回答，六 token 固定回答比较三种问题。完整交叉包含人工问答，不当自然任务表现。
- 本机所有 GPU 繁忙，按项目备用规则使用 WN-A100；共享 NAS、gia 和 CUDA 已核验。GPU 0 UUID GPU-25dab3f9-1eda-bcfd-291e-9ca1ca3b673b；远端 controller 1255、worker 1257、watcher 1256 已全部退出，GPU 归零。墙钟 198.90 秒、峰值 allocated 20286.14 MiB，无 OOM 或重试；日志 local-qa-controls/controller.log。仅使用一张卡，勿重复启动或改冻结源码。
- source-qa-controls-v1、基础 commit 2fd25b6。启动器显式执行冻结脚本，此阶段实际 runtime_source、controller 均核验来自冻结副本。4752 组距离独立重算最大误差 3.33e-16，396 组旧 Q 梯度逐值一致，792 组跨问答视觉均值特征完全相同，实际 LoRA-A 输入接口一致；全部完整主预测均为原生整体梯度。
- 结论和完整图表见 [[gradient-prototype-transfer]]：控制实验不支持把差异仅归于答案短；保持回答换问题已能触发变化，长答扩写不恢复末层 Q 判断。K/V 在公开头部稳定、腹部仍波动；后续医学迁移已完成，见本页首节，不能沿用公开结果宣称迁移有效。问题词语与有效长度、无标点答案的原生对照仍待分离。
- 逐 token 分解发现短答的句号预测方向接近整体梯度，但不证明去标点就能修复。BF16 分项加和与原生梯度最多 2.37% 相对 L2 差，个别分类也不同；该边界和仅六家族的限制已写入记录，禁止把分解当作私有攻击输入。source-qa-controls-report-v1、首版报告备份与最终 source-qa-controls-audit-v1 全保留，没有重跑 GPU 或修改主分数。
- 发布文件在 docs/experiments/data/visual-concept-probe/qa-controls-*，图已检查，只有公开聚合、无私有图文。完整 CPU 回归 461 通过、4 跳过，十二项定向与 Ruff 通过；最终文档及定向 48 项、十六份记录及本地产物核验、MkDocs 严格构建和三个源码快照、发布 SHA 均通过。保留其他文本方向与 paper 工作，不提交 Git。

## 2026-11-01 问答类型机制审计已完成

- 用户希望解释描述式与 Yes/No 的差异并寻找缓解方案。本阶段只用已保存的公开合成图梯度，未启动 GPU、读取医学目标或改变原校准与预测。完整谱、分数间隔、截断对照和原因边界已并入 [[gradient-prototype-transfer]]，不提交 Git。
- 入口 evaluation/qa_type_gradient_diagnosis.py，产物位于 outputs/diagnostics/multilayer_property_20261101/multi-qa/type-diagnosis；analysis-source 和 qa-type-manifest 保存源码、公开输入及聚合哈希。全部 396 份原公开梯度校验一致，原分数重算误差零，不重复运行。
- 事后审计曾提出监督位置、末层 Q 读出与类间间隔压缩的解释；它本身不能独立归因问法语义。之后的 7B 控制与 token 分解已在上方阶段完成，不再只依赖小型结构检查；三种简单公开谱截断未修复的旧结论保留，未据此调整医学目标。
- 当时提出的同问题、同答案、token 分解和 Q/K/V 控制现已完成，以上方最新状态与统一实验记录为准；不要按旧计划重复 launch，也不在原二十图上选择模块。
- 仅 CPU 5.03 秒；定向结构与数值五项、最终文档及定向 41 项、Ruff、十六份记录及本地产物核验、MkDocs 严格构建、发布文件与原梯度哈希均通过。保留文本方向和 paper 所有无关修改。

## 2026-11-01 公开多问答校准已完成

- 用户澄清其授权目标为“公开多问答校准”，上一轮因果接口检查不代替本阶段。实施细案已在下一阶段计划追加；结果继续汇入 [[gradient-prototype-transfer]]，不新增记录或提交 Git。
- 新入口 evaluation/public_multiqa_calibration.py；仅用六十六张公开 dev 合成图，四种校准 QA、两种留出 QA，内容家族偶数拟合、奇数审核，各风格和种子同组。原型、模型、原 private 二十图自然问答上传均保持不变；不读取目标文本、不重构 QA。
- 主 multi_ridge 与 single_ridge 同算法、同公开图片，四层标准化距离差，固定 ridge=0.1、截距不惩罚，类别及内容家族等权。另保留第 24、32 层单独阈值校准与原评分。先公开审核，再按预定规则在全部公开 dev 四 QA 重拟合，冻结后盲评两属性，预测提交后才评价真值；不据医学标签改权重或阈值。
- 根目录 outputs/diagnostics/multilayer_property_20261101/multi-qa，source-multi-qa-v1 冻结，基础 commit 2fd25b6，现有 gia。启动时本机最空闲卡为 GPU 3，UUID GPU-34cc75da-dd86-68fb-13e2-c740238bdc3b；controller 307851、worker 307879、watcher 307852 已全部退出，显存零、利用率零，未使用第二张卡。原 worker 在评分全部提交后因基线表示核对停止；source-multi-qa-finalize-v1 只接续 CPU 评价，没有重做评分或拟合，完成以 multi-qa/complete 为准，勿重复 GPU launch。
- 准备时原 fit 标签文件不含 dev 行，已改用原公开生成器声明并保留 prepare-failed 与 prepare-repair。末尾原型距离与旧对称分差的等价表示修复后，80 组单层基线换算误差为零、原 fusion 也一致；原 failed、evaluation-repair 与冻结预测均保留。首份原生五层回放误差为零，实际运行源码在修复前与快照全部一致。
- 全部数字和条件边界见 [[gradient-prototype-transfer]]：只改善公开合成腹部的留出问法，未改善医学自然问答迁移，不替换旧方法或在目标上继续调权重。原队列仍有模态混淆；下一步若继续，应在新公开条件中分开控制视觉领域和问答类型。
- 独立重算 2700 组公开逐层距离、所有规则与公开留出预测以及 90 组目标指标均通过；396 份公开梯度、原模型、输入与冻结源码哈希通过。聚合、CSV、图和 manifest 已导出 docs/experiments/data/visual-concept-probe/multi-qa-*，不含私有图文。完整 CPU 回归 456 通过、4 跳过，新增五项及 Ruff 通过；最终文档及定向 48 项、十六份记录及本地产物核验、MkDocs 严格构建和两份源码快照、发布文件 SHA 均通过。本阶段已完成，不提交 Git，保留其他方向与 paper 改动。

## 2026-11-01 视觉位置文本依赖检查已完成

- 用户要求验证公开视觉模板是否需要原始问答。本阶段仅用六张公开合成图，改变后置问答内容、长度、是否保留后缀，以及前置公开模板；逐个视觉位置比较全部三十二层 q_proj 与实际 LoRA-A 输入，并单独比较五层 LoRA-A 梯度和行空间。没有读取医学私有样本或重构文本。
- 新入口 evaluation/causal_visual_feature_check.py，根目录 outputs/diagnostics/multilayer_property_20261101/causal；source-causal-v1 冻结，基础 commit 2fd25b6，现有 gia，不提交 Git。运行计划保留原生 bf16 与原模型，数值差异按原值报告，不看结果改精度或容差。
- 本机 GPU 1 UUID GPU-31e4da09-78b1-dc97-41a3-8965bc3d3ad5，controller 273697、worker 273725、watcher 273698 全部退出；运行 247.56 秒、峰值 allocated 20076.12 MiB，显存与利用率已归零。首份产物哈希已核对，无重试或精度调整，不重复启动。
- 全部对照、前置模板与裁剪的边界见 [[gradient-prototype-transfer]]：原始问答不进入公开视觉模板，但仍影响上传；本阶段不是新的医学分类测试，不改变层级或聚合规则。保留同前缀、同槽位长度的固定公开后缀即可，勿将裁剪前向的 bf16 差异写成逐值等价。
- 独立 analysis 审计已核验冻结源码、模型、公开输入、54 份条件和对应向量，1728 组均值特征及 150 组梯度指标重算通过。聚合与清单在 docs/experiments/data/visual-concept-probe/causal-feature-*；全部 CPU 回归 451 通过、4 跳过，定向九项与 Ruff 通过。最终文档及定向 45 项、十六份记录及本地产物、MkDocs 严格构建和发布文件哈希均通过；已完成，不提交 Git，其他文本方向与 paper 改动保留。

## 2026-11-01 LM 动态候选对照已完成

- 用户授权在 Top-k 防御下先做小规模验证；计划见[动态候选](plans/2026-10-23-federated-text-pilot.md#dynamic-candidates)，结果继续汇入 [[dager-upload-defense]]。沿用原 pilot 的两条短问答、两条中等问答，不按恢复结果选择；固定四组 gradient100、gradient116、dynamic16、lm_only，不提交 Git。
- 新入口 evaluation/dynamic_candidate_study.py；完全 Residual、beam 16、LM prior 0.01、prefix batch 32、200000 次评分及 3600 秒上限不变。dynamic16 为梯度 Top-100 与当前候选前缀下的 LM Top-16 并集，重合后按梯度排序补齐到与 gradient116 相同的 125 分支；共同 ambiguous ID 保留。LM-only 仅接收公开长度和模型，不接收梯度、图像或 Observation，独立运行一次，不将两类图像重复当成额外样本。
- 三个梯度方案×两类图像×四 QA，加四次 LM-only，共 28 次重构；全部提交后才统一评价。原 gradient100 也重跑并逐份核对历史 winner；独立引擎与原搜索分数、winner 和计数一致，候选补充、LM-only 隔离和延迟评价等定向 44 项通过，Ruff 通过，生产攻击未修改。
- 根目录 outputs/diagnostics/dynamic_candidates_topk_20261101；source-v1 SHA256 ddd65ca355f2265cc7da9d18ac55e7ce6ea94631c91434982484d0bd6b947181，基础 commit 2fd25b6，现有 gia。WN-A100 GPU 0 单卡，UUID GPU-25dab3f9-1eda-bcfd-291e-9ca1ca3b673b；2026-11-01 19:47 完成，墙钟 27.04 分钟，峰值 allocated 18.47 GiB。controller 863、worker 868、CPU 评价 1076、报告 1086、watcher 873、显存巡检 875 全部退出，GPU 最终零显存零利用率。
- 动态组相对原基线的 ROUGE-L 为 GT 0.194→0.206、Random 0.148→0.173；相同预算 Gradient-116 为 0.194、0.161。所有方案 Question/Answer/Joint EM 均为 0/4；Random 一例补回两个真实 token 后，第 5 步仍被综合打分淘汰，GT 一例反而提前剪枝。当前不替换原方法，不据四条继续调 M、权重或扩大样本。
- LM-only 的四个问题解码后均为空，两个答案不含字母或数字；这是当前固定 text-only prior、特殊 token 排除及已知长度约束下的退化，不能据其零分推广通用 LM 能力或单独量化梯度贡献。GT 梯度分支还使用公开图像，LM-only 不用；没有看结果后改 prompt 重跑。
- 全部结果哈希、模型及冻结源码、八次原 baseline winner 与文本指标、GT/Random 同上传哈希、125 等分支、七组聚合及前缀/配对审计通过。八份聚合表图与审计已复制至 docs，清单 docs/experiments/data/federated-text-pilot/dynamic-candidates-manifest.json；没有原文或 token ID。68 项定向及索引测试、Ruff、十六份记录及本地产物核验、MkDocs 严格构建通过；三轮共二十二份聚合文件 SHA256 一致。不重复启动、不提交 Git，其他图像工作与 paper 修改保留。

## 2026-11-01 固定问答属性对照已完成

- 用户授权的同二十图固定 QA 配对诊断已完成，完整层级与属性结果、模态分组、配对区间和聚合图见 [[gradient-prototype-transfer]]，没有新增独立样本或拆分实验页。固定任务不替代未知原自然问答任务结果。
- 根目录 outputs/diagnostics/multilayer_property_20261101，独立 fixed-qa、local-fixed-qa 与 source-fixed-qa-v1；公开模板、尺度、阈值、模型和原图均沿用，原自然上传与结果未改。source-fixed-qa-report-v1 和报告 manifest 保留，已通过全部源码哈希和二十图配对重算审计。
- 本机 GPU 1 UUID GPU-31e4da09-78b1-dc97-41a3-8965bc3d3ad5，controller 195702、worker 195756、watcher 195703 全部退出，GPU 已释放。墙钟 9.50 分钟，峰值 allocated 20527.31 MiB，无 OOM 或自动重试；现有 gia 和基础 commit 2fd25b6，不提交 Git。
- 固定任务为 text_known Observation，通用公开 QA 不透露部位；实际分类接口只接收上传张量、公开模板和规则。一份自然上传的全部分数复现误差为零；客户端核对原像素，预测保存后按私有 run 映射配对，改变的训练数据仅为 QA。
- QA 内容、有效长度和监督任务一起变化，患者关系未知，部位与模态混淆未解除；第 32 层仍是预定次要条件，主多层没有据结果替换。任何下一步确认需新数据和公开规则，不在这二十图上调阈值。
- 完成标记 fixed-qa/complete、analysis-complete 与 integrity-audit；完整 CPU 回归 443 通过、4 跳过，定向九项测试及 Ruff 通过。其他文本防御和 paper 工作保持原样。

## 2026-11-01 谱加权文本筛选已完成

- 用户授权在既有三十 SLAKE QA 上验证谱加权 token filter，计划见[谱加权筛选](plans/2026-10-23-federated-text-pilot.md#spectral-defense)。保留完全 Residual、同一份 FedSGD B1/Noise/Top-k 上传和 GT/Random 条件，beta=0、0.01、0.1、1 预先固定，lambda=beta×median(s_i²)，不依据真值调参。
- 新入口 evaluation/spectral_filter_study.py；四档共 720 组分数全部提交后才统一评价，informative K=100 与共同 ambiguous 保留规则不变，只测候选召回、FullCoverage、Informative AP，不做序列搜索。直接岭回归目标一致性、原端点复现等定向 37 项测试及 Ruff 通过；未改变生产攻击、环境、原始上传或 paper，不提交 Git。
- WN-A100 GPU 0 单卡，UUID GPU-25dab3f9-1eda-bcfd-291e-9ca1ca3b673b；GT/Random 串行。controller 643、GT 648、Random 756、CPU 评价 804、报告 817 均成功结束；十分钟 watcher 653、三十秒 NVML 显存巡检 655 已安装。其他卡均有任务，没有使用或停止。
- 根目录 outputs/diagnostics/spectral_filter_20261101，冻结 source-v1 SHA256 8da55ae61b2b0bb03e6fd5d46eb4dacb0deffac4524af980ab3ae06441bbc11a，基础 commit 2fd25b6，使用现有 gia。2026-11-01 19:02 完成，墙钟 3.75 分钟、峰值 allocated 17.05 GiB；720 组分数统一评价及表图已保存。180 组零加权端点与原候选精确一致，最大分数差 6.85e-14；所有 beta 实际候选均为 109，不重复启动。
- 当前四档谱加权没有恢复 Noise 或 Top-k 的完整候选，FullCoverage 均为 0/30；强加权还把无防御完整覆盖从各 26/30 降至 GT 25/30、Random 24/30。Noise 的谱变平，s_min/s_max 约 0.585，单靠按强弱降权不足以修复方向污染；不替换原方法，不据当前队列继续选 beta。结果见 [[dager-upload-defense]]。
- 冻结源码、模型、全部 180 份分数哈希、先评分后真值、共同预算、二十四组指标重算和十八组配对审计通过；七份聚合表图及审计已复制至 docs，清单 docs/experiments/data/federated-text-pilot/spectral-manifest.json。清单命名冲突已修复并恢复旧清单，两轮十四份聚合文件 SHA256 核验一致；过程见 setup/export-repair.json。保留所有图像方向和 paper 修改，不提交 Git。
- 全部实验 PID 已退出，显存巡检自动退出，成功后收尾尚在十分钟等待的 watcher 653；最终 GPU 0 为零显存零利用率，见 setup/process-finalization.json 及最终远端核验。Ruff、37 项定向测试、24 项索引测试、十六份记录及本地产物核验和 MkDocs 严格构建通过，无需重复启动。

## 2026-11-01 多层属性与腹部测试已完成

- 用户授权的本机测试已全部完成，结果现见 [[gradient-prototype-transfer]]（2026-11-02 拆分）。旧二十图 CPU 聚合、新二十图自然问答上传的 Head 与 Abdomen 推断、配对与模态分组均已保存；研究结论以记录为准。
- 早先将沙箱内 nvidia-smi 失败误判为整个会话无 GPU；用户追问后使用 require_escalated 验证并直接运行，无需 SSH。入口 examples/run_multilayer_property_local.sh；execution 只在本机单卡，未在其他服务器代跑。
- 根目录 outputs/diagnostics/multilayer_property_20261101。生成 controller/worker/watcher 109491/109545/109492；probe 128451/128480/128452，全部退出。本机物理 GPU 1 UUID GPU-31e4da09-78b1-dc97-41a3-8965bc3d3ad5 已释放，进程内 cuda:0；source-generation-v1、source-probe-v1、source-report-v1 和所有 provenance 保留，不重复启动。
- 生成四十八张非医学腹部外观图，核验保留三十九图；发现实际只有八个重复提示家族，在任何受害模型评分前改为四个拟合、四个开发家族，原图不改、不额外生成。另有四十八张公开头部与肢体负例；修订与限制见 review，不把编号隔离误报为提示家族隔离。
- 新队列五 MRI 头部、九 CT 腹部、一 MRI 腹部、五胸部 X-Ray，排除六十三个旧属性目标 ID；患者关系未知且模态混淆。探针只接收匿名 Observation 与公开模板，先提交预测再评估标签，head 沿用公开规则，abdomen 仅用公开开发估计尺度；不按新标签改层或权重。
- 最终完成标记 local-probe/complete、fresh/complete 和 fresh/analysis-complete。生成 6.15 分钟，probe 阶段 10.93 分钟，记录峰值 allocated 20526.45 MiB。完整 CPU 回归 435 通过、4 跳过，Ruff 通过；文本防御、paper 及所有无关修改保留，未提交 Git。

## 2026-11-01 部分公开输入消除对照已完成

- 用户授权的 alpha=0、0.5、0.9、1 对照已于 18:17 完成，墙钟 8.89 分钟；三十 QA、三种上传、两类图像共 720 组筛选分数全部提交后统一评价，未做序列重构。结果见 [[dager-upload-defense]]，计划见[部分消除对照](plans/2026-10-23-federated-text-pilot.md#soft-residual-defense)。
- 固定相同上传和候选预算，梯度与 token 特征都使用 I-alpha UUᵀ；informative K=100，所有 alpha 共用 hard residual 的九个 ambiguous ID，实际候选数均为 109。180 组 alpha=1 与历史候选完全一致，最大分数差 6.85e-14；Raw 的候选额外保留规则与旧原生 Raw 不同，不混比。
- 四档中完全 Residual 在六组防御和图像条件的平均召回与 Informative AP 均最高；Noise 与 Top-k 的 FullCoverage 在所有权重下仍为零。当前不支持减弱消除权重，不按该队列继续调更细 alpha；下一步若获授权，保留完全 Residual 优先验证防御下的子空间估计或缺失坐标处理。
- 根目录 `outputs/diagnostics/soft_residual_filter_20261101/`；source-v1 SHA256 `2618782dccc6fbb8494b67d87a6be5d6f38945617b273ed18f8eba96b33f891c`，基础 commit 2fd25b6；现有 gia，bf16 模型与原上传，float64 分析。独立评估入口没有改变生产攻击，环境、原上传、图像工作和 paper 均未改；本阶段未提交 Git。
- WN-A100 最初未挂载 NAS，没有启动 GPU；用户恢复后核验共享目录、模型及 CUDA。GPU 5 启动前被其他任务占用，实际只用 GPU 0（UUID `GPU-25dab3f9-1eda-bcfd-291e-9ca1ca3b673b`）串行 GT/Random，未干扰其他任务。
- controller 360、GT 365、Random 498、CPU 评价 536、报告 549 已成功退出，单卡本轮占用已释放；十分钟 watcher 370、三十秒显存巡检 372 也已退出；远端进程表与 GPU 显存均已核验。不重复启动或自动扩展任务。
- 全部 scores 及结果在 NAS；七份聚合表图及审计复制到 docs，清单在 docs/experiments/data/federated-text-pilot/soft-residual-manifest.json。源码、模型、180 份分数哈希、共同预算、先评分后真值及二十四组总体指标重算通过；GPU peak allocated 16.06 GiB。
- 定向 DAGER 回归 31 通过，Ruff、十六份实验记录及本地产物核验通过，文档索引测试 24 通过，MkDocs 严格构建通过。Informative AP 排除共同 ambiguous ID，不能与包含这些 ID 的 AP 混比；结果不代表完整序列恢复，也不宣布独立验证的最优 alpha。

## 2026-11-01 文本防御退化诊断已完成

- 用户要求解释与 llm_privacy_eval 相同防御参数下的退化差异，并允许必要的小验证；本阶段复用原三十样本上传的既有分数、奇异谱及四份 Enron 四十样本报告，仅执行 CPU 事后审计，无新 GPU、模型前后向或重构。结论和完整数字已追加 [[dager-upload-defense]] 的上传防御章节，没有新建实验页。
- 新入口 evaluation/defense_text_diagnosis.py；根目录 `outputs/diagnostics/defense_text_diagnosis_20261101/`，audit.json 与 audit-vectorized.json 内容完全一致。240 组原 K=100 候选精确复现；五档固定 K 只评价已有全词表排序，没有按真值改变候选评分或选择新输出。
- 防御公式、参数优先级、逐张量 Top-k 含义已核对；完整问答 EM 与 PII 实体召回不可直接比较。公开输入消除后的归一化残差很弱，噪声残差显著增加；这是残差范数诊断，不是已测得的文本信噪比，也不是控制全部跨项目差异的因果实验。
- 聚合及 SHA256 在 docs/experiments/data/federated-text-pilot/defense-diagnosis.json 与 defense-diagnosis-manifest.json；只有聚合，无私有 token 或原文。私有 token ID 只在原重构全部提交后由 evaluation 用于覆盖评分，原攻击、参考仓库、环境和 paper 均未改动。本阶段代码和文档未提交，保留图像方向全部无关修改。
- 本机会话没有 GPU 直通，本机 SSH 入口也未完成主机密钥验证；未绕过校验，也未因此改用 WN-A100 启动新任务。当前 CPU 证据足以定位第一阶段排序问题；后续如获授权，应先固定模型、精度和上传匹配有效扰动，再验证防御下候选评分，不先扩大 beam。
- 收尾：Ruff、十六份实验记录及本地产物核验通过，文档索引测试 24 通过，MkDocs 严格构建通过；两份审计内容完全一致，文档聚合、来源及脚本 SHA256 核验通过。

<!-- BEGIN next-stage-live -->
## 2026-10-28 下一阶段最新状态

- 凸组合恢复及公开开发已完成，结果现见 [[convex-token-recovery]] 与 [[candidate-scoring]]（2026-11-02 拆分）。
- 根目录 `outputs/diagnostics/image_property_next_20261028/`；单卡顺序运行、各阶段源码冻结，详见 plan、各 launch、progress 与 complete 标记。已知受控问答的公开机制 pilot 不能替代真实医学目标或恢复文本主结果。
- 第一版生成保留，固定一次质量修订后核验最终一百零八图；所有新预测先保存再加入查询标签。C、PCA、自由 tokens 与独立真实目标确认没有执行。
<!-- END next-stage-live -->


## 2026-10-28 下一阶段实验已完成

- 本轮 A/B、历史二十图深层诊断和十二公开查询 D1 已完成，完整结论现见 [[gradient-prototype-transfer]] 与 [[convex-token-recovery]]；执行计划仍使用[下一阶段计划](plans/2026-10-28-image-property-next-stage.md)。C、PCA、自由 tokens 与新真实目标确认未执行。
- 根目录 `outputs/diagnostics/image_property_next_20261028/`，基础 commit 2fd25b6。新增评估代码、聚合与图尚未提交；paper 未动。源码 source-development-v1/v2、source-deep-probe-v1、source-readout-v2 与 source-report-v1/v2/v3-statistics 及所有失败、取消记录保留。
- 优化 worker 23190、controller 22951、其 watcher 22952，CPU 诊断 23315/23316，只读接续 controller 23633/23634 及实际 readout PID 均已退出。本轮单卡 WN-A100 GPU 1 UUID `GPU-6a309ac1-08fa-062a-b449-e8de39ba4d09` 已释放；巡检时该卡随后有其他进程，不停止或复用它，任何新实验重新核验资源。
- 以 `postprocess-pipeline/complete.json`、`convex/complete.json`、根目录 `report_complete.json` 和 `deep-probe/complete.json` 为最终状态，不能只按旧 failed 标记判断整体状态。原属性读出 dtype 错误后从完整系数接续，没有重跑优化；首次采集元数据及准备脚本导入路径错误也已显式修复并留档。
- 上传匿名化并一次性打乱，映射仅在评估端。预测与连续分数先提交，后读取真实标签；新图只用本地 Qwen 生成并独立核验，不外发私有问答，未修改核心训练和攻击方法。
- 预定主层、次要层、后验预算审计和系数投票分开报告，不据旧医学标签替换主方法。下一步先做文本场景分组与公开选择迁移，暂不扩大 tokens 优化规模。
- 收尾核验：完整 CPU 回归 423 通过、4 跳过，Ruff、十六份记录及本地产物检查、MkDocs 严格构建通过。错配对照区间按六组共享输出的上传对修正，原聚合保留备份，系数、预测与点估计不变。

## 2026-10-28 图像分析独立分支提交

- 按用户最终安排，main 已提交既有文本实验代码 270eb81、文本结果与网页站点 77b8b54；分支 `experiments/image-property-ideas-20261028` 从该 main 创建，只增加今天的图像分析代码、测试、计划、报告与聚合表图。
- 图像代码提交 2573d8e，统一结果仍见 [[image-property-ideas]]；图像分支的共享首页、索引和本交接只增加对应图像内容。main 的文本与站点提交不包含今天的新图像分析结果，不推送远端。
- 论文草稿 `paper/` 与 `docs/paper/` 留在工作区；原始 outputs、私有样本与模型不纳入 Git。代码隔离定向测试二十七项通过，最终分支完整 CPU 回归 411 通过、4 跳过；Ruff、十六份记录及本地产物检查、MkDocs 严格构建通过。报告与图的原始数值及哈希不变。

## 2026-10-28 文本实验与站点提交

- 按用户要求，既有文本实验代码、记录与网页站点修改提交到 main；今天的图像分析使用独立分支。代码提交 270eb81 包含 AdamW 回放数值适配、显式差值提案和固定 QA 数量的 SGD 对照，原研究边界不变。
- 文本结果见 [[federated-text-pilot]]；站点新增 KaTeX 数学渲染和一级 Plans 入口。原始运行产物留在 outputs；论文草稿不纳入本次提交，也不推送远端。
- 分支隔离副本中代码回归四百项通过、四项跳过；索引重建后全部三十六项文档测试通过，Ruff 通过。图像工作未混入 main 的实验目录与导航。

## 2026-10-28 图像属性下一阶段规划

- 已整合用户补充分析与既有结果，形成[下一阶段唯一计划](plans/2026-10-28-image-property-next-stage.md)。原始计划保留历史设计与冻结设置；后续执行细案并入新计划，结果继续归入 [[image-property-ideas]]，不拆分结果页。
- 先验证跨风格统计与视觉 tokens 优化的二阶链，再按门槛推进受约束恢复；深层探针先核对空间与位置，文本引导生成作为场景线索充分时的补充。单域失败、模态鸿沟与深层梯度可读出之间的因果关系仍待验证。
- 本轮仅写计划并补充入口，没有修改攻击实现、生成图像或启动 GPU 实验；该计划不改变文本防御鲁棒性的既有研究优先事项。

## 2026-10-28 图像属性文档归并

- 按用户要求，本轮 idea 验证统一为 [[image-property-ideas]]；[[synthetic-property]] 仅保留原合成候选基线。加权、类内抑制、概念探针及接口读出的结果、限制与运行记录均移入统一记录，原始数值与产物位置不变。
- 计划仅保留[图像属性方案](plans/2026-10-28-image-property-ideas.md)，将评分与概念探针两个执行细案并入其末尾，移除两个重复计划页及独立概念实验页。下面的阶段完成条目都指向同一份实验记录。
- 当前汇总脚本只导出 outputs 中的阶段草稿，不再自动覆盖旧报告或重建拆分实验页；冻结的历史源码与完成清单保持原样。文档重组前备份和映射见 `outputs/diagnostics/documentation_reorg_20261028/`，没有新 GPU 实验。
- 迁移核验保留全部原始表格行，两个阶段的报告和聚合哈希均未改变；十六份实验记录与产物核验、四十六项相关测试、Ruff、MkDocs 严格构建通过。

## 2026-10-28 视觉概念探针检查已完成

- 当前结论与完整对照见 [[visual-concept-probe]]，执行设计见[视觉概念检查](plans/2026-10-28-image-property-ideas.md#concept-design)。D1 预定主条件未过门槛；E1 接口一致且真实视觉 tokens 有属性读出能力，二者不能混为梯度恢复成功。未启动 D2 梯度探针、E2 自由 tokens 优化或文本引导生成。
- 只用九十六张公开合成候选与原模型，不读取私有目标、恢复问答或上传；全部预测保存后才加入公开标签。根目录 `outputs/diagnostics/visual_concept_probe_20261028/`，source-v1、source-v2、首次失败记录与完整性产物均保留；代码提交 2573d8e，聚合与报告随图像分支提交。
- WN-A100 GPU 1 UUID `GPU-6a309ac1-08fa-062a-b449-e8de39ba4d09`，启动前为最空闲卡，使用现有 gia；controller 20932、worker 20935、watcher 20933 均已退出，GPU 本轮占用已释放。日志 `outputs/diagnostics/visual_concept_probe_20261028_v2.log`，最终标记 predictions_complete、report_complete 与 finalized，不重复启动。
- 首次检查因评估 loss 归约与原生计算有 1.43e-6 差异而停止，logits 和梯度均为零误差；修正归约后保留相同科学条件及容差，四个接口样本均完全一致。五项针对性测试、Ruff、十六份记录及产物核验、MkDocs 严格构建通过，无 OOM 或模型修改。

## 2026-10-28 图像属性评分验证已完成

- 当前结果与配对区间见 [[candidate-scoring]]；执行方案见[评分验证](plans/2026-10-28-image-property-ideas.md#scoring-design)。公开候选选择的张量加权尚未显示相对等权评分的稳定增益。
- 根目录 `outputs/diagnostics/property_geometry_20261028/`；GPU 5 首份缓存经核验后，按用户新授权迁移至空闲 GPU 6。source-v1、source-v2-gpu6、迁移与完整性记录均保留；代码提交 2573d8e，聚合与报告随图像分支提交，不覆盖旧结果。
- 全部预测提交后才评价目标标签；回放已结束，本轮 GPU 占用已释放。下一步先核对主比较，再决定是否用新目标确认；不自动启动概念探针或视觉 tokens 优化。

## 2026-10-28 图像属性推断思路整理

- 按用户要求整理[图像属性推断方案](plans/2026-10-28-image-property-ideas.md)，涵盖类内变化抑制、张量判别加权、文本引导生成、视觉子空间概念探针，以及此前讨论的视觉 tokens 恢复；每项包含假设、验证实验与推进条件。
- 建议先验证候选库内判别加权，再测试软抑制与硬投影的域外效果；直接词向量探针先检查几何对应。类内变化不等于纯外观噪声，文本消除后的 LoRA 梯度行空间只被包含在视觉残差空间中。
- 本轮完成文档和理论核对，没有实现新攻击或启动 GPU 实验；该计划不改变此前文本防御鲁棒性的优先事项。当前属性实验结论仍见 [[synthetic-property]]。

## 2026-10-26 文本恢复优先研究防御鲁棒性

- 用户明确将防御条件下的文本恢复退化列为当前优先问题；先研究加噪与稀疏化下的候选漏检，再考虑序列搜索优化。全局裁剪保留作对照，既有结果见 [[dager-upload-defense]]。
- 下一步建议先固定单样本、更新方式和精度，建立防御强度与候选完整覆盖的关系；分别验证噪声下的子空间估计、Top-k 缺失坐标处理。防御感知方法须声明其可用信息，在开发数据上确定规则后独立验证。
- 本轮只核对既有实验、评分实现和已保存的子空间诊断；未运行新攻击、修改攻击算法或启动 GPU 任务。AdamW 适配与批量搜索对照保留为后续方向，不自动扩展矩阵。

## 2026-10-26 固定四 QA 的 SGD 划分对照已完成

- 用户授权的 B1S4、B2S2、B4S1 已于 12:27 完成；21 份捕获、42 次 GT/Random 重构全部 completed，无预算耗尽或 OOM，墙钟 147.82 分钟。统一评分、总体及长度表图和收尾核验已完成，当前结果见 [[federated-text-pilot]]，计划见[固定四 QA 阶段](plans/2026-10-23-federated-text-pilot.md#fixed-four-qa)。不要重复启动旧任务。
- 从原冻结 SLAKE 三十 QA 预先排除短桶和中桶各自最后一条，保留 28 QA，短 9、中 9、长 10；全部原客户端 0，保持剩余顺序组成七组。三条件分组及组内顺序完全相同，每份上传四条不同 QA、各访问一次；只有七组独立上传，不作稳定步数趋势判断。
- 使用原 Round-10 模型，bf16 主干及计算、FP32 LoRA 参数、梯度及 SGD 差值；lr=0.01，无 momentum/weight decay，cast cache 与 TF32 关闭。K=100、beam=16、LM prior=0.01、prefix batch=32、rerank=0；每份上传 800000 次评分、14400 秒。固定学习率时累计更新幅度及 loss 加权也变化，不能解释为纯步数效应。
- 根目录 `outputs/diagnostics/fixed4_multistep_slake28_20261026/`；source-v1 SHA256 `7a7437639c07b900443d8c7232bffca1a9f8de5db885404fb8bad8fba4d0e567`，基础 commit aba0a9f。gia 环境不变，本机准备 NAS，远端执行；源码、模型、分组与预算全程不变，新增代码和聚合产物未提交，paper 未动。
- WN-A100 controller 9550、capture 9555、GT 9718、Random 9721、评分 10775、报告 10789、十分钟 watcher 9557、显存巡检 9559 全部退出。GPU 2/5 已由本项目释放；GPU 5 随后被另一个进程占用，不终止或复用它，任何新任务均重新核验资源，最多两卡。
- 三条件 GPU 原生 SGD 合成检查均逐项一致；21 份真实捕获的后续步骤使用更新后参数且初值未变。GT/Random 上传配对一致，Random 无公开图像；攻击仅看公开 Observation，全部结果提交后才评分，未按真值调方法或分组。
- 所有标准 attacks 路径已切回 NAS，不依赖 WN-A100 临时盘；九个自动报告 SHA256、42 个最终检查点、六组总体与十八组分桶指标重算通过。十一份聚合表图和审计副本清单在 `docs/experiments/data/federated-text-pilot/fixed4-manifest.json`，收尾脚本在 setup/finalize-audit.py；只用既有结果，没有新增 GPU 实验。
- CPU 代码回归为 401 通过、4 跳过，Ruff 通过；本轮只追加收尾审计和文档。下一步如获授权，优先在固定候选和预算下检查 N=4 的联合序列搜索，或用独立分组验证趋势；不自动扩大矩阵或重启 AdamW。
- 收尾文档索引及本地产物检查通过（15 份记录），索引测试 24 通过，MkDocs 严格构建通过；十一份聚合副本及来源 SHA256 全部一致，无新 Git 提交。

## 2026-10-26 AdamW 四样本 pilot 已完成

- 用户授权的五条件、十份上传、二十次 GT/Random 重构于 00:24 完成，全部 completed，无预算耗尽或 OOM；统一评分、表图、候选上界审计和公开上传反变换诊断已整理，结果见 [[federated-text-pilot]]，计划见 [AdamW 执行计划](plans/2026-10-23-federated-text-pilot.md#fedvlmbench-adamw)。
- 目录 `outputs/diagnostics/adamw_text_pilot_20261025/`，source-v1 指纹 `3c7cbf286c6f4e8b334069254e3da10df802a57d18c11e9c02959fa9d3d264b5`，基础 commit aba0a9f，gia 环境不变；新增代码仍未提交，paper 未动。CPU 全量 394 通过、4 跳过，Ruff 通过。
- WN-A100 controller 9199、capture 9204、GT 9351、Random 9354、watcher 9206、显存巡检 9208、评分 9503、报告 9513 全部退出；GPU 2/5 显存为零，所有标准 attacks 路径归档 NAS。不重复启动旧任务，不自动扩大矩阵。
- FP32/bf16 两种参数及状态的 GPU 原生双步 smoke 均与实际 torch.optim.AdamW 完全一致；每个本地步骤使用更新后参数，上传内保留 moments、每份上传从零初始化。lr=1e-4；只替换原 Round-10 模型的本地更新，不是完整 FedVLMBench 重训。
- AdamW 差值子空间只作启发式提案，完整优化器重放选择十六个候选，共 320 次重放。候选上界只是全部重构提交后的私有参考评分，没有向攻击者提供真值或重新选择实际输出；反变换诊断只读取公开上传，没有新 GPU 任务。
- 后续判断：当前不值得直接扩展到三十 QA；先解决单步 FP32 AdamW 的候选生成，再验证多步。AdamW 直接反变换已有数值病态，不能未经验证直接套公式；四样本零恢复不代表 AdamW 提供隐私保证。
- 来源清单 `docs/experiments/data/federated-text-pilot/adamw-pilot-manifest.json` 含十份聚合副本；九个自动报告和全部聚合 SHA256、总体指标重算均通过。原始结果与私有样本仅留在 outputs。收尾：15 份实验记录及本地产物检查通过，文档索引测试 24 通过，MkDocs 严格构建通过；无新 GPU 任务或 Git 提交。

## 2026-10-25 仅代码修改已提交

- 用户明确要求只提交代码、排除 paper。commit `aba0a9f`（Add pooled DAGER evaluation and SGD precision controls）包含十八个源码、配置及测试文件；覆盖 pooled DAGER、防御未知评估、Gaussian noise、标准 SGD 数值路径及精度实验入口。
- 用当前已提交基线加暂存补丁构造独立副本，完整 CPU 回归 382 通过、4 跳过，Ruff 通过，全部暂存文件与受测副本逐字节一致。未推送。
- 文档、实验结果和图、`docs/paper/` 与 `paper/` 全部保留在工作区，未纳入该 commit。实验记录已登记代码 commit，但文档及聚合产物仍为 uncommitted；后续提交继续遵循用户指定范围。AdamW 四样本 pilot 随后已完成，见本页最新状态。


## 2026-10-25 混合精度真实多步对照已完成

- 原三十 QA 的混合精度矩阵已于 23:19 完成，45 份重新捕获的上传、90 次 GT/Random 攻击全部 completed，无预算耗尽或 OOM；全部重构保存后统一评分。当前结果、表图与限制见 [[federated-text-pilot]]，计划检查清单已完成。
- 目录 `outputs/diagnostics/mixed_multistep_slake30_20261025/`；源码 `source-v1` 指纹 `529566c53e748be0941f0720c73b15a112e232f09f62be6becfbcebf748e28e3`。原 Round-10 bf16 数值保持，仅 LoRA 参数提升至 FP32；capture/attack 均使用 bf16 autocast、FP32 LoRA 与 SGD 上传、关闭 cast cache，每步新前后向。不是全模型 FP32，也不是复用固定旧梯度。
- WN-A100 controller 8132、capture 8137、GT 8295、Random 8298、watcher 8139、显存巡检 8141、评分 9116、报告 9138 全部退出，GPU 2/5 显存为零；controller、watch-latest、memory-latest 都为 completed。不要重复启动，后续新实验需重新核验空闲卡且最多两卡。
- 十五份双步轨迹、源码与精度、派生模型指纹、配对上传全部核验通过；六组总体指标重算通过，九个自动报告及十二份文档聚合副本 SHA256 一致。全部标准 attacks 路径切回 NAS，不依赖远端临时盘；收尾审计见 `setup/finalization-audit.json`，文档副本清单见 `docs/experiments/data/federated-text-pilot/mixed-manifest.json`。
- 既有全 FP32 阶段因速度按用户要求停止，保留旧目录 `outputs/diagnostics/fp32_multistep_slake30_20261025/`、45 份捕获、2 次完整攻击及在途检查点；不是 OOM，不报告完整恢复率。原 controller 7651、worker 7803/7806 与两种巡检均已退出；不继续旧矩阵。
- 收尾文档校验：15 份记录及本地产物检查、24 项索引测试和 MkDocs 严格构建通过；当前无新增运行任务。
- 后续 FedVLMBench AdamW 四样本诊断已完成，见本页当前状态；默认 DAGER 仍不直接支持 AdamW，显式差值提案诊断必须重排且不保证梯度子空间性质。代码已提交为 `aba0a9f`，文档及聚合产物仍未提交。


## 2026-10-25 AdamW 舍入小验证已完成

- 用户授权对照 FedVLMBench 的 AdamW lr=1e-4；复用十五组公开梯度与原 Round-10 六个 LoRA-A 块，在本机 gia 上做单步 CPU torch.optim 算术，零动量、betas=(0.9,0.999)、eps=1e-8、weight_decay=0。无新 GPU、训练或重构，原混合精度矩阵继续运行，设置未改。
- 结果与适用范围见 [[federated-text-pilot]]；目录 `outputs/diagnostics/adamw_rounding_slake30_20261025/`，state 与 summary 均 completed，脚本 `setup/run.py`。FP64 首步公式验证及原 SGD bf16 零比例复现通过，聚合 JSON/CSV 与 SHA256 已复制文档；没有 Git 提交。收尾：15 份实验记录和本地产物检查通过，文档索引测试 24 通过、MkDocs 严格构建通过，聚合副本与来源 SHA256 一致。

## 2026-10-25 单步 SGD 精度对照已完成

- 用户授权的标准 SGD 修正与固定梯度精度对照已于 2026-10-25 18:41 完成，九十次新攻击全部 completed，无预算耗尽或 OOM，统一评分与报告已生成。当前结论及完整表图见 [[federated-text-pilot]]；本页已升为 key，track summary、headline 与首页核心结论已同步。
- 固定同一份 bf16 梯度，理想差值与 FP32 单步更新的 Joint EM 均回到原梯度基线；标准 bf16 SGD 仍归零，说明本轮主要损失来自低精度参数舍入。原乘法路径已修正，但修正本身没有恢复效果。不能将此结果推广为多步 FedAvg 或 FP32 前后向训练的结论。
- 根目录 `outputs/diagnostics/sgd_precision_slake30_20261025/`；controller、watcher、memory-latest 均为 completed。controller 6877、watcher 6881、prepare 6851、worker 6892、6895、显存巡检 7034、评分 7526、报告 7538 均退出；不要重启已完成任务。全部攻击已镜像 NAS，不再依赖远端临时盘。
- 固定源码 `source-v1` 指纹 `a2c3e9d3f0d4e5907a78510832f320ae4312e2e7ae7667ca1a40b3200d0cfdd5`，原 gia 环境未改。CPU 标准 SGD 生成公开上传，GPU 2、5 执行文本攻击；原 30 样本结果和源码保留，当前 core/fl.py 使用 torch.add(..., alpha=-lr)，保留可微回放。本轮没有 Git 提交。
- 首先通过 CPU 回归 378 项、跳过 4 项，CUDA 上修正函数式与标准 SGD 的 bf16/FP32 算子对照通过。攻击结束后补做十五组所检六个 LoRA-A 块的 CPU/CUDA 上传对照：FP32 逐坐标完全一致，bf16 为 99.9473%；仅执行张量算术，无重训或重跑攻击。结果在 `setup/cpu-cuda-control.json`。
- 收尾核验：15 份记录及本地产物通过、文档索引测试 24 通过、MkDocs 严格构建通过；八份新增文档聚合文件 SHA256 通过，GPU 2、5 显存均为零。
- 新报告 `report/final-comparison.json` 明确旧两条件复用、新三条件运行；文档聚合副本、图及 SHA256 在 `docs/experiments/data/federated-text-pilot/precision-manifest.json`。九份自动报告文件哈希通过，恢复结果已统一评分。
- 补充分辨率审计见 `setup/update-resolution.json` 与实验页：所检 LoRA-A 的逐坐标相对更新中位数为 0.009375%，93.2743% 更新低于半个相邻 bf16 间隔，与实际零差值逐坐标完全对应。未重跑攻击；聚合副本已加入 precision-manifest.json（现九份文件）。
- 后续真实混合精度单步与双步矩阵已完成，见本页当前状态和 [[federated-text-pilot]]；批量独立搜索和中长文本剪枝诊断仍为待办。

## 2026-10-25 SLAKE 30 样本扩展已收尾

- 用户授权的完整 SLAKE 队列扩展已于 2026-10-24 02:39 完成，2026-10-25 核验并整理。165 份上传、330 次攻击均为 completed，无预算耗尽；统一评分与表图已生成。结论、总体与分桶结果见 [[federated-text-pilot]]，后续优先做单步数值与批量搜索对照；不自动开启新实验。
- 2026-10-25 补充单步数值审计已完成，见 [[federated-text-pilot]] 的多步更新章节：十五组所检六个 LoRA-A 块的原梯度全部非零，bf16 差值平均仅 6.71% 非零，逐坐标符合当前代码；另发现函数式 `param - lr*grad` 与标准 SGD `add_(alpha=...)` 的低精度路径不完全一致。先做理想差值、标准 SGD、FP32 master 分离对照；本轮零恢复不能解释为 FedAvg 或步数本身有效保护。运行代码尚未改动，无新增 GPU 实验。
- 收尾检查：15 份实验记录及产物通过，文档索引测试 24 通过，MkDocs 严格构建通过，十一份文档聚合副本 SHA256 通过。
- 运行根目录 `outputs/diagnostics/federated_text_slake30_20261023/`，`controller.json`、`report-complete.json` 与 `watch-latest.json` 均为 completed。controller 6732、watcher 6749、worker 6737、6740、评分 6803、报告 6813 全部退出，GPU 1、2 已释放；不要重启旧调度器。
- `setup/completion-audit.json` 确认源码、GT/Random 配对上传及暂停前 44 份结果哈希一致；报告九个文件的 SHA256 核验通过。`report/` 保存完整聚合表图；文档副本以 `slake30-` 命名，来源清单为 `docs/experiments/data/federated-text-pilot/slake30-manifest.json`。旧四样本结果保留，实验页已改用 30 样本口径。
- 冻结源码 `source-v1` 指纹仍为 `eeb1c28574d7a0ce6f8b3199c621e6d2dbe78838363786068a6d1cc5770480e3`，模型为既有 LLaVA-1.5-7B rank 32 Round 10，LM prior=0.01。源码与环境没有因 NAS 故障改变；本轮代码、聚合产物尚未提交，不自动提交。
- NAS 阻塞后曾迁移临时检查点到 WN-A100 `/tmp/giavlm-slake30-20261023-io`；用户暂停后又显式续跑，保留原累计预算。全部攻击已镜像 NAS，各 run 的 `attacks` 已切回 `attacks-nas`，不依赖远端临时盘。恢复脚本和审计保存在 `setup/io-fix/`、`setup/resume-1/`，不要将运行耗时作为严格速度对比。

## 2026-10-23 场景扩展验证已完成

- 用户授权的小规模验证已完成，计划见 [联邦文本 pilot](plans/2026-10-23-federated-text-pilot.md)，结论、完整表图和限制见 [[federated-text-pilot]]。本轮代码与聚合产物尚未提交，不自动提交。
- 固定 SLAKE 四条问答、LLaVA rank 32 Round 10，22 份上传、44 次配对攻击与统一评分全部完成，无预算耗尽；源码与配对上传哈希核验通过。离线回归 130 通过、2 跳过，补充配置与索引检查 27 通过，Ruff 通过。
- WN-A100 controller 5350、GT worker 5312、Random worker 5354 已完成退出，本项目 GPU 占用已释放。产物 `outputs/diagnostics/federated_text_pilot_20261023/`，`controller.json` 与 `state-evaluate-all.json` 为 completed；不要重复启动。十分钟 watcher 5351 已收尾退出，`watch-parallel-latest.json` 为 completed、44/44；收尾索引测试 24 通过，15 份记录及产物检查通过，MkDocs 严格构建通过。
- 后续三十样本扩展、单步精度对照与真实混合精度双步矩阵均已完成，见 [[federated-text-pilot]]；批量 beam 分支竞争仍待诊断。
- 源码固定在 `source-v1`，使用原 `gia` 环境；本机准备文件，远程执行 GPU，最多两卡。参考仓库 `llm_privacy_eval` 保持只读，环境未修改；原有 `docs/paper/`、`paper/` 未动。

## 2026-10-18 已完成阶段

- 更新时间：2026-10-18（Asia/Shanghai）；分支 `dager-rank-and-experiment-records`（自 `73ebac2` 起）已快进合并回 `main` 并推送到 `origin/main`；分支保留，与 `main` 同步。
- 用户已明确要求提交现有内容。2026-10-11 至 10-18 的代码按研究主题拆分提交，相关实验页已登记代码 commit，方法文档与聚合图同步整理；原始产物继续保存在本地 `outputs/`。
- `lm_prior_weight=0` 仍是默认值。提交不表示将先验、属性评分或生成候选切换为新的默认攻击；原版 `dager` 仍未实现。
- 本轮五个新 commit 均在干净检出中逐一验证：全量 CPU 回归最终为 355 通过、4 跳过，Ruff 通过；文档相关测试 36 项通过。14 份实验记录及本地产物检查通过，MkDocs 严格构建通过。

接手顺序：`AGENTS.md` → 本文件 → `docs/experiments/README.md`。开始前检查 git、进程和 GPU，不能以调度器中的 `running` 字样判断任务存活。

## 本轮提交

| Commit | Content |
|---|---|
| `9efa65d` | 可选纯文本 LM prior、配对实验入口与测试 |
| `b304158` | 重构文本与 LM prior 文本的图像可区分性诊断 |
| `76e7c70` | 公开原型库属性推断、平衡扩展与对照 |
| `92169af` | 本地 Qwen 图像生成测试入口 |
| `561eeb2` | 合成 CT 候选的部位属性实验 |
| `a8282b3` | 混合模态头部属性实验、生成修复与测试 |
| `2440189` | Qwen3.5 adapter 与 DAGER 线性注意力输入 |
| `6beafa7` | 跨模型文本恢复 benchmark 与只读巡检 |
| `e2ce836` | 通用摄影候选库的头部属性对照 |
| `eb0c5a4` | 仅限 loopback 的文档在线编辑器、首页与索引卡片、中文搜索 |
| `cedd460` | 合并实验记录、记录结构校验、benchmark 实验页与 WN-A100 规则 |

此前筛选研究 `7be400e`、图像排序诊断 `de95a9f`、rank 对比 `4bae2c8` 和日志修复 `e9a05eb` 已提交。`cedd460` 将 `dager-multisample-n10`、`dager-lora-rank` 合并为 [[dager-multisample]]，将 Qwen 图像生成、合成 CT 与头部属性三页合并为 [[synthetic-property]]；旧 id 已不可链接。历史展示材料位于 `docs/ppt/dager_branch_summary_20261011_v2/`，保留本地，不纳入 Git。

## 当前结论

- **文本恢复**：Residual 改善了第一阶段 token 排序，但候选完整不保证序列恢复。rank 对比见 [[dager-multisample]]，筛选研究见 [[dager-filter-phase]]。
- **纯文本先验**：rank 32 的冻结权重验证有收益；相同权重在 rank 8 收益有限，仍有候选漏检和早期剪枝。成功问答的重复度限制了结论，见 [[dager-lm-prior]]。
- **图像可区分性**：正确文本能支持部分全局排序，但局部像素优化仍失败；重构文本的内容与顺序都影响排序，LM prior 验证增益仍存在不确定性，见 [[gradient-discriminability]]、[[text-unknown-discriminability]]。
- **图像属性推断**：公开库三类平衡扩展显示相对 Model-only 的信号，但相对校准文本对照及同部位泛化的优势仍不明确，见 [[image-property-inference]]。
- **合成候选**：医学外观候选支持头部属性诊断，但三套通用人体摄影库未保留该效果，正确文本也未恢复表现。历史样本重叠、患者相关性、候选外观差异及生成模态偏差仍限制结论，见 [[synthetic-property]]。
- **图像重构**：IG 仍未得到有效像素恢复；从近真值起点出发也可能漂移，不能把候选排序成功解释为图像重构成功，见 [[ig-failure-diagnosis]]。

## 下一步优先级

- **最新完成：固定 N=4 的 SGD 划分对照**。B1S4、B2S2、B4S1 的七组、42 次配对重构和统一表图已完成；见本页当前状态及 [[federated-text-pilot]]。后续优先诊断候选完整但序列失败，或独立分组验证；不自动启动新任务。

- **待做提醒：FedVLMBench AdamW 多步实验**。用户已要求列入待做；四样本 pilot 已完成并汇总，候选提案失败；当前优先考虑单步 AdamW 第一阶段的适配，不扩大矩阵，不重复启动。计划见 [联邦文本实验计划的 AdamW 待做项](plans/2026-10-23-federated-text-pilot.md#fedvlmbench-adamw)。采用 AdamW 初始 lr=1e-4、bf16 前后向、FP32 LoRA 参数与优化器状态、FP32 上传差值；先适配并验证当前只支持 plain SGD 的 pooled DAGER，再做单步和多步 GT/Random 对照。纯 bf16 参数及状态作为精度消融；现有 AdamW 小验证只有单步 CPU 算术，不能替代文本重构。本次仅登记，不启动新任务。
- **已完成的执行计划**：[跨模型文本恢复 benchmark](plans/2026-10-17-crossmodel-text-benchmark.md) 的检查清单全部完成。结果与表图见 [[crossmodel-text-benchmark]]；不自动开启额外实验。
- **跨数据集与长度**：三个数据集的冻结分桶、图像隔离、文本重叠与原生长度均已审计，72 个 QA 的配对矩阵已完成。两个新数据集的长桶各 n=1，不能据此声明稳定趋势。
- **实验入口**：`evaluation/text_benchmark_data.py` 准备与冻结数据，`evaluation/text_benchmark.py` 执行配对捕获与攻击，`evaluation/text_benchmark_report.py` 生成聚合表图；Qwen 使用独立 `gia-qwen35` 环境，旧 `gia` 保留。
- **本轮训练口径**：每个模型与数据集组合独立训练到 Round 10，再单独捕获单样本 FedSGD 更新。主实验使用 rank 32、alpha 64 与 LM prior 权重 0.01；跨域冻结模型和更多方法消融不属于本轮必需矩阵。
- **长度控制**：按实际内容 token 而非 padding 槽位计长，分别报告问题与答案。固定容量需预留 EOS；禁止静默截断后仍按原文长度解释结果。
- **文本方法**：保留现有 token filter。rank 8 优先诊断候选完整但真前缀被剪的样本的几何分差与 NLL 分差；rank 32 的最终排序失败单独保留，不用验证集反复调参。
- **属性方向**：通用摄影库对照已完成，先检查候选领域与外观差异，再决定是否利用可恢复的文本线索生成领域候选；见 [[synthetic-property]]。后续仍应在独立开发数据上冻结评分与阈值，不按当前目标分数筛选候选或调整到全对。
- **其他待办**：TAG 的公开图像与问题条件仅有 pilot；IG 的旧参数扫描使用 bf16 候选，引用前需按当前 float32 主副本重跑。相关边界见 [[tag-text-only-public-image]]、[[ig-parameter-sweep]]。

## 运行状态与模型

- 2026-10-18 通用摄影候选库对照已完成，见 [[synthetic-property]]。用户明确允许共享本机 GPU 5，UUID `GPU-8ad9e9b4-e159-4f78-7036-2d1d018e7b7d`。根目录 `outputs/diagnostics/generic_head_n20_gpu5_20261018/`，`generation-source`、`runtime-source` 固定代码；复用原二十张目标的观测与重构文本，未重新训练或搜索文本。生成 worker 2123789、watcher 2123790 和评估 worker 2142475、watcher 2142476 均已退出；七十二张生成图全部保留，120 份预测、输入与模型一致性校验完成，`complete.json` 和 `report_complete.json` 均已保存。本次 GPU 已释放，不要重复启动；评估代码已提交为 `e2ce836`，原医学库结果未覆盖。
- 用户已批准跨模型、跨数据集实验，完整步骤、固定参数与恢复规则见上述计划。本项目全部任务合计最多同时占用 WN-A100 的两张空闲物理卡，GPU 编号与 UUID 启动前重新核验。
- WN-A100 已通过 SSH 别名 `zx-wn-a100` 验证可达，具备 8 张 A100 80GB；共享仓库、`gia` Python 和 Qwen3.5-4B 权重均可访问。后续可以从当前会话经 SSH 执行实验，连接与运行规则已写入 `AGENTS.md`；启动前重新检查远端 GPU 占用与运行时兼容性。
- 本轮 benchmark 已完成，完整结果与边界见 [[crossmodel-text-benchmark]]。六个组合的十轮训练及 288 次配对攻击均完成，全部攻击为 completed；六组 summary、报告来源 SHA256 与 ZIP CRC 已核验。原 `gia` 未变更，代码和新环境均在本机准备，GPU 程序只在 WN-A100 两张卡上执行。代码已提交为 `2440189`、`6beafa7`。
- CPU 阶段调度器原 PID 835、watcher 836，`controller.json` 为 completed、`all_gpu_jobs_complete`；后续不要根据历史启动记录重复运行已完成任务。
- 本机 CPU 汇总原 PID 1507970，`finalizer.json` 为 completed。完整表图位于本轮根目录 `report/`，打包为 `report-aggregate.zip`，含 18 个聚合文件，无私有示例；文档内聚合副本与 SHA256 清单位于 `docs/experiments/data/crossmodel-text-benchmark/`，四张完整矩阵图已更新。`interim/slake/` 仅为历史阶段输出。
- 用户要求的每 10 分钟后台巡检已正常退出，原 PID 1525580；`monitor/timer.json` 为 finished，完成 31 次检查，最终 health 为 completed。最新报告为 `monitor/latest.md`，历史为 `monitor/history.jsonl`；没有下一次检查，不需要重启已完成实验的监控。
- 用户要求的会话主动监控也已执行至实验完成；最后巡检为 2026-10-18 03:18，`monitor/live-session.json` 记录 completed、288/288。监控期间未自动重试 OOM、改预算或改实验条件。
- 工作区盘点与 SLAKE 初步审计保存于 `outputs/diagnostics/workspace_audit_20261017/`。Q64、A64 容量变化使旧 SLAKE 快照不再直接复用，本轮六个组合均需重新训练。
- `gia-qwen35` 已在本机创建于共享 NAS，安装 Transformers 5.20.1、PEFT 0.21.2、Accelerate 1.15.0，保留 Torch 2.6.0；新环境 `pip check` 通过，旧 `gia` 的 pip 快照前后一致。本轮产物根目录为 `outputs/diagnostics/crossmodel_text_20261017/`；训练使用冻结 `source-smoke-v1`，正式攻击使用 `source-benchmark-v1`。数据准备完成：`data/*/frozen.json` 共冻结 72 个 QA，正式配对矩阵为 288 次攻击；VQAv2 固定训练与评估调度使用的 816 张图片已校验。
- 2026-10-18 检查时，本机没有本项目的运行进程；此前其他任务 PID 1151928 已不在进程表中。
- rank 8 快照：`outputs/federation/slake_llava_f_l_fedavg/round-0010/`。
- rank 16、32 快照：`outputs/diagnostics/dager_rank_gpu5_20261010_v2/models/rank-16/round-0010/`、`rank-32/round-0010/`。
- GPU 环境为 Conda `gia`，CPU 测试为 `/tmp/giavlm-venv`；文档仅在本地查看。长任务必须 detached 启动并配 watcher，首个 capture/checkpoint 验证通过后才能离开。

## 复现与本地数据边界

- Observation 当前为 schema 5，模型 overlay 为 schema 4。DAGER checkpoint 为 schema 2；旧 checkpoint 只用于只读审计，不静默迁移续跑。
- rank 32 先验验证观测的轮次计数曾标为 0，但实际模型哈希对应第十轮；F-L 梯度逐项对照不受影响，原始观测未改写，后续捕获已修正，详见 [[dager-lm-prior]]。
- 私有图像、完整重构文本及样例图继续留在 `outputs/`。CT 讲解与用户要求的头部属性六图对照，通过 `docs/experiments/examples/` 下的本地片段和符号链接展示，该目录已忽略；页面使用可选片段，Git 中只保存说明与聚合结果。六图对照见 [[synthetic-property]] 的图像直观对比，真图及两类候选按固定记录顺序选择，未按推断结果挑选。
- `GIA/`、其他参考项目、数据、模型和展示压缩包均按既有规则忽略。不要将生成图的模态提示直接当作真实模态标签。
- 本轮文档页将 `uncommitted` 标记改为 false，并记录相应代码 commit；这不意味着原始本地数据或大型产物被纳入 Git。
- 小型聚合结果按实验保存在 `docs/experiments/data/` 并纳入 Git，包含对应本地来源的 SHA256；不包含逐样本文本、图像或原始样本标识。

---
重要提交、实验阶段结束或方向改变后更新本文件。具体数字与运行记录写入实验页，链接为 `[[id]]`；保持测量结果、假设与后续计划的区分。
