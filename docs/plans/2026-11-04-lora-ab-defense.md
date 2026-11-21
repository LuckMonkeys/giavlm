# LoRA A/B 耦合与防御下的 token 筛选

用户授权验证 A/B 一致性和公开输入消除后的左子空间，先写计划，再执行实验；结束后结果并入 [[dager-upload-defense]]。只验证第一阶段，不运行序列搜索、不改生产攻击、不提交 Git。代码与环境在本机 NAS 准备；本机与 WN-A100 合计最多使用两张空闲 GPU，本阶段优先只用本机空闲 GPU 2。

## 研究问题

- 单步 FedSGD 的 bf16 上传在多大程度上满足 A/B 恒等式？公开输入消除是否放大一致性误差？
- 从另一份 B 梯度估计的残差左空间是否有可靠低秩结构？GT 与 Random 是否不同？
- 在同上传、同候选预算下，一致性校正或左空间投影能否改善 Noise 下的 Candidate recall、Informative AP 和 FullCoverage？

## 理论与边界

对产生上传时的公开模型权重 A、B，令有效权重为 W+cBA，干净梯度满足 G_A Aᵀ=Bᵀ G_B。这里 A、B 是第十轮快照，不是最早初始化；c 同时出现在两份梯度中而抵消。

令 P 是公开输入的正交投影，Q=I-P，R=G_A Q，则

    C_obs = Bᵀ G_B_obs - G_A_obs P Aᵀ
    C_clean = R Aᵀ

只有当 R 有低秩结构、A 的右侧压缩保留了这些方向时，C 的左空间才有助于过滤 R 的噪声。Random 的视觉分量消除不精确，不能把私有文本长度上界当作严格秩保证。全部 32 维被保留时投影是恒等操作；没有可靠方向时不应把梯度强行置零。

恒等式使用原始梯度尺度；全部方法最后沿用原始观测 G_A 的归一化因子，避免混入额外块加权。只比较同一次 FedSGD 捕获，不套用到 AdamW 或多步参数差值。本阶段主条件为 None 与 Noise；Top-k 的缺失坐标不符合此 Gaussian 校正模型，不在本阶段扩展。

## 固定设置

| Item | Setting |
|---|---|
| Model | 原 SLAKE LLaVA-1.5-7B、F-L、rank 32、alpha 64、Round 10，原 bf16 模型与上传 |
| Source | outputs/diagnostics/federated_text_slake30_20261023/，原 30 QA、短中长各 10 条；不重新训练、捕获或加噪 |
| Uploads | 原 fedsgd-b1 与 noise_ratio=0.01；GT/Random 使用同一上传，评分只接收各自声明的公开图像或 seed=0 替代图 |
| Modules | 第零层 Q/K/V，每块使用匹配的公开 A、B 权重及观测 G_A、G_B；不引入额外层、语言先验或词汇规则 |
| Analysis | float64；原 SpanFilter 的 numerical rank、vocab chunk 和歧义保留规则不变 |
| Budget | informative K=100，加共同 ambiguous IDs，预计每组候选均 109；120 份上传图像组合×4 个方法=480 组筛选分数 |
| Evaluation | 全部 480 组分数保存并核验后才读取参考 token；报告 Candidate recall、Informative AP、FullCoverage，以及分长度与配对变化 |
| Execution | 预检本机 GPU 2 UUID GPU-ce1827e8-5b97-c17f-66da-3acf01a97692 空闲；WN-A100 GPU 2/7 也可用但不同时启动本阶段任务；实际使用卡数为一，启动前复查 |
| Output | outputs/diagnostics/lora_ab_filter_20261104_v2/；source-v2 固定，gia 环境不变；首轮失败目录另行保留 |

Gaussian gate 是防御感知的机制诊断：噪声比例 0.01 是预先固定的实验假设，不是从当前 Observation 推断的值。原 Observation 不提供防御参数，不能把此对照称作原 defense-unaware 攻击的等知识结果；A/B consistency 与未加门槛的 Residual left 不需要该比例。GT/Random 的私有图文边界保持原条件。

## 四个预设对照

| Method | Rule |
|---|---|
| Baseline | 原完全 Residual，逐上传复现历史分数与候选集合 |
| AB consistency | 联合最小二乘投影到 X Aᵀ=Bᵀ Z；同时修正两份观测，不将有噪声的 B 梯度当作精确真值。相对方差权重取各自观测最大绝对值的平方，通过 r×r Sylvester 方程求解；共同噪声比例在权重比中抵消 |
| Residual left | 对原始 C_obs 求左奇异向量；保留 k=min(公开问题与答案内容长度之和,32,数值秩)。数值秩阈值为 1e-10×最大奇异值；该长度截断在 Random 下仅作为待检验启发式 |
| Residual left gated | 在上一规则上增加观测噪声门槛；只保留超过预设 Gaussian null 最大奇异值第 95 百分位的方向。0 维或满 32 维时返回原块并记录，不悄悄抹去整个梯度 |

Residual left 在 k=0 或 k=32 时也显式返回原块，避免浮点恒等投影引入伪差异。四组相互独立地处理原观测，不串联叠加。

### Noise gate 的公开尺度

- Noise 使用 sigma_A=0.01×max(abs(G_A_obs))、sigma_B=0.01×max(abs(G_B_obs))；这是公开观测代理，不是私有干净最大值或已知精确方差。
- 64 次固定种子的 Gaussian null 模拟，不采集新梯度。Bᵀ N_B 的行协方差由 BᵀB 给出，N_A P Aᵀ 的列协方差由 A P Aᵀ 给出；用两个 r×r 正半定平方根模拟，保留变换后的有色噪声结构，不直接套用白噪声 MP 边界。
- None 条件不假设存在 Gaussian 防御噪声，gated 退化为同一长度截断，只作数值与损伤控制。
- 不按 token 真值选择秩、噪声门槛、模拟种子或方法；若门槛全部拒绝，报告无可利用左方向及原块回退比例，不再扫描更低门槛。

## 分阶段执行与停止条件

- [x] 数学测试：精确恒等式、带权联合投影、残差 sketch、0/满秩回退、历史分数复现和先评分后真值边界。相关测试 39 项通过，Ruff 通过。
- [x] 冻结源码及输入哈希；本机 GPU 2 detached 启动，watcher 与显存巡检，核验首份产物。source-v2 首份候选精确复现、分数最大差 1.81e-14，四方法分数和 A/B 一致性均通过检查。
- [x] 120 组公开观测的四方法评分及原模型未修改核验；每块保存原始/残差一致性误差、奇异谱、选秩和噪声门槛。
- [x] 全部分数提交后统一评价；再做配对干净上传的梯度误差和左空间诊断。此额外干净上传只属于事后参考，不进入 noisy 评分或参数选择，单独标记其使用边界。
- [x] 整理聚合表图、来源与 SHA256，扩展既有 docs/experiments/2026-10-23-dager-upload-defense.md，更新交接、索引并核验资源释放。定向测试 39 项、文档索引测试 24 项、Ruff、21 份实验记录及本地产物、MkDocs 严格构建和 16 个发布/代码哈希均通过。

本轮结束条件是四种预设方法全部测试完成，或出现阻断性数值/资源失败并明确记录；不自动 OOM 重试、改 dtype、换预算或扩展 Top-k。若 Random 无完整覆盖改善，或只有 GT 有收益，结论限定为机制对照，不宣布有效未知图像改进，不继续扩大搜索或提交代码。

## 运行记录

启动前已检查 Git、历史任务、NAS 空间和两机 GPU。工作区含既有图像、文本、防御与 paper 修改，全部保留；本轮新增独立 evaluation 入口和测试，不改生产 SpanFilter。

- 2026-11-04 本机 detached 启动；controller PID 2030591、GT worker 2030596、watcher 2030627、显存巡检 2030629。GPU 2 显式按 UUID 绑定，进程内为 cuda:0；GT/Random 串行，不使用 WN-A100。
- Interpreter：/home/zx/nas/miniconda3/envs/gia/bin/python；base commit 2fd25b6，source-v1 SHA256：35569d21ddef0fd8fdf239aa21dfa67b43a914b4e2305c82bef51c4202a9b334。
- Launcher：bash outputs/diagnostics/lora_ab_filter_20261104/setup/launch.sh；worker 命令、PID、时间与日志在该目录的 controller.json、setup/score-gt.log、setup/score-random.log。后台 watcher 600 秒写一次完整状态，异常或结束即时写终态并退出；显存巡检间隔 30 秒。
- 首轮在评分前发现设备错误：load_observation 保留 CPU 上传张量，必须显式移至 CUDA 投影基所在设备。失败目录与冻结源码不覆盖，未读 token 真值，controller/watcher 已终止。
- 修复后目录 outputs/diagnostics/lora_ab_filter_20261104_v2/，source-v2 SHA256：5ef098123fc6f87a321443471f96a684bf99aacb56b0b064d5a3fd546de7ad16；仅修正设备放置，全部条件保持。
- source-v2 detached controller 2037999、GT worker 2038004、watcher 2038035、显存巡检 2038037；仍只绑定本机 GPU 2，旧任务全部退出。修复后定向回归 39 项再次通过，Ruff 通过。
- Random worker 2042508、CPU 评价 2046195、报告 2046795。修复后墙钟 306.65 秒、GPU peak allocated 15.83 GiB；全部 worker、controller 和监控已退出，卡上仅 4 MiB、利用率零。
- 独立审计通过：120 组原候选精确复现、最大分数差 3.25e-14，240 个评分及诊断张量哈希通过；全部方法候选均为 109，聚合与配对指标重算一致。

## 验证结论

- 三种耦合规则均已测试，没有改善 Noise 下的 FullCoverage（GT/Random 均为 0/30）；未进入序列搜索，不报告新的 Joint EM。
- 一致性校正把代数误差降至约 1e-15，但 Random 召回仅 0.1792→0.1841，GT 反而下降；不替换现有筛选器，也不提交代码。
- 左空间投影降低残差矩阵误差，但压缩 sketch 仍受噪声与相减误差影响，弱 token 方向没有得到足够保护；Gaussian gate 多数块选中零维并回退，余下块只选中一维。
- 完整表图、限制和来源记录在 [[dager-upload-defense]] 的 A/B耦合；聚合产物为 docs/experiments/data/federated-text-pilot/lora-ab-*，图为 docs/experiments/figures/federated-text-pilot/lora-ab-filter.png。
