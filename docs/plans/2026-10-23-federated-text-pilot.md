# 联邦更新与上传防御的文本恢复小规模验证

<a id="fixed-four-qa"></a>

## 已完成：固定四条问答的划分对照（2026-10-26）

用户确认补充 B1S4、B2S2、B4S1。每份上传固定同样四条不同 QA，各访问一次；每种条件复用相同分组和组内顺序，不扩展 AdamW。

| Item | Setting |
|---|---|
| Model | 原 SLAKE LLaVA-1.5-7B，rank 32、alpha 64、Round 10；不新增训练 |
| Cohort | 原冻结三十 QA，预先去掉短桶和中桶各自排序最后的一条；28 QA，短 9、中 9、长 10 |
| Grouping | 保留剩余原顺序，连续四条一组，共七组；全部来自原客户端 0；三个条件分组完全相同 |
| Conditions | B1S4、B2S2、B4S1；每种条件七份上传、十四次 GT/Random 重构 |
| Precision | 冻结主干 bf16；LoRA 参数、梯度及 SGD 更新 FP32；bf16 autocast，关闭 cast cache 与 TF32 |
| Optimizer | 普通 SGD，lr=0.01，无 momentum、weight decay；上传 FP32 参数差值 |
| Local steps | 每步使用更新后参数重新前后向；每份上传从同一初始模型出发 |
| Attack | DAGER-Erasure-LM，K=100、beam=16、LM prior=0.01、prefix batch=32、rerank=0 |
| Budget | 每份四 QA 上传 800000 次评分、14400 秒；三种条件一致，beam 不变 |
| Knowledge | GT 使用 image_known；Random 使用 private；均声明公开逐样本内容长度 |
| Evaluation | 42 次重构全部提交后才读取参考；一对一匹配，报告 Joint EM、FullCoverage、Recall、ROUGE 和长度分桶 |
| Execution | 本机 NAS 准备代码；WN-A100 GPU 2、5 执行，最多两张；gia 环境不变 |
| Output | outputs/diagnostics/fixed4_multistep_slake28_20261026/ |
| Git | 不提交、不推送；保留 paper 及其他修改 |

固定学习率下，三种划分同时改变更新次数、累计更新幅度及原生 loss 的 batch 加权；本阶段检验同一本地数据量下的整体影响，不声称隔离纯步数因果效应。28 QA 只有七份独立上传；K=100 及跨样本联合 beam 的限制保持可见。

- [x] 确认冻结队列客户端归属、固定排除规则及七组划分；未读取恢复结果进行选择。
- [x] 新增独立执行入口；三条件离线 native SGD 精确对照通过；定向 24 项通过。
- [x] CPU 全量 401 通过、4 跳过；Ruff 通过。
- [x] 冻结源码、生成 selection.json、记录 source 指纹与模型来源。
- [x] GPU 原生三条件公开合成检查和首份真实捕获通过。
- [x] 21 份上传、42 次配对攻击及统一评分完成。
- [x] 整理三条件总体及长度表图、更新实验记录与交接。

本阶段于 2026-10-26 12:27 完成，墙钟 147.82 分钟。21 份捕获与 42 次攻击全部 completed，无预算耗尽或 OOM，统一评分和九份自动报告已生成。B1S4、B2S2、B4S1 的 Joint EM 为 GT 4/28、5/28、4/28，Random 1/28、1/28、3/28；四步仍泄露，GT 没有单调趋势，只有七组独立上传。

controller 9550、capture 9555、GT 9718、Random 9721、评分 10775、报告 10789、watcher 9557、显存巡检 9559 全部退出；本项目释放两卡，GPU 5 随后被其他进程占用，不终止或复用该进程。所有 attacks 已切回 NAS；九份报告哈希、最终检查点、六组总体与十八组分桶指标重算通过。十一份文档聚合副本及 SHA256 见 fixed4-manifest.json。源码 SHA256 `7a7437639c07b900443d8c7232bffca1a9f8de5db885404fb8bad8fba4d0e567` 不变，未提交 Git。

GPU 2 UUID `GPU-8cb10863-d4c8-528f-3add-2632ae9cb39d`，GPU 5 UUID `GPU-cb472061-1b41-cdbd-991d-51fd2749dd50`。启动前重新检查空闲与 UUID，调度器遇占用等待，不更换设备。controller、每十分钟 watcher 和无 CUDA 上下文的三十秒显存巡检均在远端 detached 执行；日志、PID 与状态写入本阶段 setup 和 controller.json。

用户于 2026-10-23 确认先做小规模验证。复用 SLAKE 的 LLaVA rank 32、Round 10 模型，检验大 batch、多步 SGD 与上传防御是否能正确运行，以及文本恢复是否退化。代码暂不提交。

## 固定设置

| Item | Setting |
|---|---|
| Model | LLaVA-1.5-7B, F-L, rank 32, alpha 64, Round 10 |
| Dataset | SLAKE，复用跨模型 benchmark 的冻结 eval 样本 |
| Cohort | 冻结顺序中的前两个 1–16 token 样本和前两个 17–32 token 样本；短、中交错组成两组 |
| Selection | 在读取本轮重构结果之前固定；不按既有恢复成功与否挑选 |
| Attack | DAGER-Erasure-LM, LM prior 0.01, K=100, beam=16 |
| Knowledge | GT 使用 image_known；Random 使用 private；均公开逐样本内容长度 |
| Optimizer | 本次捕获使用 SGD, lr=0.01；与 Round 10 前的训练优化器区分 |
| Budget | 每条样本 200000 次评分、3600 秒；双样本上传预算乘二，beam 宽度不变 |
| Evaluation | 全部重构结果保存后才读取参考文本；按完整问答文本距离做一对一 Hungarian 匹配 |
| Resources | WN-A100 GPU 2、3；GPU 3 一度被占用，释放后恢复 GT 与 Random 并行 |
| Environment | NAS 上既有 gia，环境和代码准备均在本机完成 |

## 实验矩阵

| Condition | Algorithm | Batch | Steps | Unique samples per upload | Uploads | Images |
|---|---|---:|---:|---:|---:|---|
| fedsgd-b1 | FedSGD | 1 | 1 | 1 | 4 | GT, Random |
| fedsgd-b2 | FedSGD | 2 | 1 | 2 | 2 | GT, Random |
| fedavg-b2-s1 | FedAvg | 2 | 1 | 2 | 2 | GT, Random |
| fedavg-b1-s2 | FedAvg | 1 | 2 | 2 | 2 | GT, Random |
| clipping | FedSGD | 1 | 1 | 1 | 4 | GT, Random |
| noise | FedSGD | 1 | 1 | 1 | 4 | GT, Random |
| sparsify | FedSGD | 1 | 1 | 1 | 4 | GT, Random |

总计 22 份上传、44 次攻击。每个条件都使用同样四条问答。GT 与 Random 接收完全相同的上传值。

- 裁剪：全局 L2 裁剪，统一阈值 C 为四份基准上传范数中位数的一半。阈值在攻击前冻结；这是同一小样本队列上的标定，不能解释为可部署的隐私机制。记录实际裁剪比例和相对 L2 扰动。
- 噪声：逐张量高斯扰动，标准差为该张量最大绝对值的 0.01 倍，与参考实现的 noise_ratio 定义一致。固定 seed，GT 与 Random 共用同一次噪声实现。没有隐私预算或 DP 保证。
- 稀疏化：每张量按绝对值保留 top 10%，沿用本项目 ceil 取整，原形状零填充。参考项目采用 floor 取整；该细节不直接照搬。

## 实现与理论边界

- 本项目已经支持多样本梯度和多步本地更新，但既有 DAGER 入口仅支持单样本、单步、无防御；通过显式 pooled_updates 与 defense_mode 选项开启本轮扩展，不改变默认设置。
- 多样本视图合并所有声明公开输入，使用整个上传计算共享子空间；beam 依次填充所有样本，跨样本时重置语言上下文，累计搜索分数。不读取私有 ID、文本或原始图像，也不把更新拆成单样本梯度。
- 对 batch 梯度，线性层梯度仍是各 token 外积之和。增加样本使有限 LoRA 行空间需要区分更多 token，原有 K=100 也可能成为瓶颈；本轮保持 K 固定，记录每样本候选覆盖与上传并集召回。
- 单步普通 SGD 满足参数差值 Δθ=-ηG，理想精度下与梯度具有同一行空间。单步 FedAvg 对照用来检查减法、量化等数值影响。
- 理想精度下，多步普通 SGD 的上传是不同权重处梯度之和；实际 bf16 还包含逐步舍入。第 0 层输入在 F-L 下固定，后层输入会漂移；使用初始模型做第二阶段重放属于近似，不声明精确恢复保证。
- 单份上传的全局 L2 裁剪只乘正标量。归一化后的行空间理论不变，裁剪是数值对照。加噪与稀疏化会改变行空间。
- 防御测试为 defense-unaware：攻击只看被变换后的上传，未进行去噪或防御逆变换。禁用防御条件下的全更新重排。
- FedAvg 两步使用两条不同样本，各访问一次；没有把两次访问同一条样本解释为两个独立目标。后续固定客户端数据集、增加 local epochs 是另一组实验。
- 同一批次与分步更新的原生 loss 归一化也可能不同，不把全部差异归因于权重漂移。此轮只攻击一个客户端的上传，不声称完成多客户端聚合攻击。
- 联合 beam 在样本边界不丢弃已完成样本的备选分支，可能减少后续样本的搜索多样性。本轮先保持固定实现；独立上下文下可以利用评分可分离性改进，不能把当前批量结果当作纯粹的信息量上限。

## 参考实现核对

参考仓库 `/home/zx/nas/GitRepos/llm_privacy_eval` 只读。

- `core/model_dager.py:compute_grads_fed_avg` 区分本地数据量、b_mini、avg_epochs；上传给攻击的是 -Δθ/(lr×steps)。普通 SGD 下可解释为平均梯度，AdamW 下不能这样解释。
- `get_matrices_expansions` 对 FedAvg 的估计秩额外增加 60。这是经验规则，本轮不移植。
- `defenses/gaussian_noise.py` 使用逐张量 max-absolute 缩放，已作为独立 gaussian_noise 防御实现；不复用本项目会同时裁剪的 gaussian_dp。
- `defenses/gradient_clipping.py` 使用全局 L2 范数；`gradient_compression.py` 使用逐张量 top-k。

## 执行进度

- [x] 核对两仓库实现、协议与默认支持边界。
- [x] 固定小规模样本选择和七条件矩阵。
- [x] 实现可选多样本搜索和显式 defense-unaware 路径。
- [x] 完成离线回归与真实权重首次捕获检查：130 通过、2 跳过；448 个 bf16 上传张量；GT 与 Random 的哈希一致，Random 没有公开图像。
- [x] 冻结源码并在 WN-A100 启动，保存 PID、UUID、命令、日志：controller 5245，watcher 5246，capture 5249；源码 `source-v1`。
- [x] 完成 44 次攻击后统一评估；全部 completed，源码与配对上传哈希一致。
- [x] 整理聚合表图、实验记录及交接；本轮没有提交代码。

运行产物计划目录：`outputs/diagnostics/federated_text_pilot_20261023/`。运行状态、失败原因与完整启动信息放在该目录，不在此记录中存放私有样例。

首次捕获检查还确认四条问答属于同一个已分配客户端。完整审计保存于 `setup/validation.json`；本机准备代码、远程执行，既有环境未改动。

22 份更新已捕获完成。两次攻击预检分别遇到 GPU 利用率残留和 GPU 3 被其他任务占用，均在攻击开始前停止，原记录保存在 `controller-capture.json`、`controller-preflight-2.json`。实际攻击改由 GPU 2 串行执行，controller PID 5306、watcher PID 5307；攻击条件、预算与固定源码均不变，不重跑捕获。

GPU 3 随后释放，改由 controller 5350 接管原 GT worker 5312，并在 GPU 3 启动 Random；watcher 5351。原串行调度器主动退出，记录保存在 `controller-serial.json`，GT 进程没有中断。巡检新增 PID 一致性检查，当前状态看 `controller.json` 与 `watch-parallel-latest.json`。

2026-10-23 本轮完成。controller 5350、两名 GPU worker 均已退出，统一评分成功；完整结果见 [[federated-text-pilot]]。聚合表图在 `report/`，文档副本及 SHA256 在 `docs/experiments/data/federated-text-pilot/`；不自动扩展矩阵。

## SLAKE 30 样本扩展（2026-10-23）

用户随后明确要求使用之前 SLAKE 的完整 30 样本设置，不再只测试四条问答。本节是新增授权阶段，上面的四样本结果保持不变。

| Item | Setting |
|---|---|
| Cohort | 原跨模型 benchmark 冻结的全部 30 个 SLAKE QA，三个长度桶各 10 个，30 张不同图像 |
| Pairing | 保留 frozen.selected 原顺序，相邻两条组成双样本上传；已核验全部属于同一客户端 |
| Model | 原 SLAKE LLaVA-1.5-7B Round 10，rank 32、alpha 64、Q64、A64；不重新训练 |
| Attack | 与 pilot 相同；LM prior=0.01、informative K=100、beam=16、rerank=0 |
| Precision | 保留 bf16 参数更新；没有混入 FP32 数值对照或新搜索方法 |
| Defenses | 固定 pilot 的 C=2.0968051343155176、noise_ratio=0.01、top-k ratio=0.1；不按 30 样本结果重调 |
| Images | GT 与 Random 配对，逐样本内容长度公开 |
| Matrix | 原七条件；四个单样本条件各 30 上传、三个双样本条件各 15 上传，共 165 上传、330 攻击 |
| Budget | 每样本 200000 evaluations、3600 秒；双样本上传乘二；不自动重试 OOM |
| Evaluation | 全部重构保存后统一评分；总体 n=30，并按原三个长度桶分别汇总 n=10 |
| Artifacts | outputs/diagnostics/federated_text_slake30_20261023/；原四样本目录不覆盖 |
| Runtime | WN-A100 最多两张空闲卡；准备阶段 GPU 2、3 空闲，启动前再次检查；共享 gia 环境不改动 |

- [x] 核验完整冻结队列、分桶和客户端分组；未按重构结果筛选。
- [x] 扩展实验入口，固定 clip 阈值，保留旧 pilot 默认行为；新增长度分桶汇总和聚合报告。
- [x] 定向回归 134 通过、2 跳过；Ruff 通过。
- [x] 冻结源码，启动 detached controller 5434、十分钟 watcher 5435、capture worker 5439；首份捕获核验通过。
- [x] 完成 165 份捕获、330 次攻击和统一评估，自动生成表图；全部 completed，无预算耗尽。
- [x] 完整结果已更新到既有 [[federated-text-pilot]]，核验全部进程退出；没有提交 Git。

新结果尚未产生，不能将四样本恢复率当作 30 样本结果。完整长度覆盖和双样本分组与 pilot 有变化，不能将两轮差异只归因于样本量。

30 样本阶段的首份捕获已核验：448 个 BF16 上传张量，GT 与 Random 的更新文件 SHA256 一致，Random 没有公开图像。`setup/first-capture-audit.json` 记录 passed；当前仍在捕获阶段，330 次攻击尚未完成。配置与文档索引检查通过，MkDocs 严格构建通过。

### 运行中 I/O 调整（2026-10-23）

- 165 份 capture 已完成。GPU 3 被其他任务占用后，用户明确授权使用 GPU 1；当前为 GPU 2 的 GT worker 5537 与 GPU 1 的 Random worker 6008。
- 新 controller 6003、watcher 6010；原调度器退出，GT worker 未重启。调度脚本与迁移审计位于新产物目录 `setup/io-fix/`。
- 高频检查点写入改到 WN-A100 本地 `/tmp/giavlm-slake30-20261023-io`；每轮后台同步保存最新完整检查点，完成结果逐文件哈希核验后写回 NAS。临时历史代次回收，原 NAS 历史全部保留。
- 运行中的 `runs/*/attacks` 可能指向远端本地盘，本机读取时需使用 `runs/*/attacks-nas` 中的已同步副本；不要把本机无法解析远端链接误判为结果丢失。一个上传的两种图像攻击完成后，正常 `attacks` 路径会自动切回 NAS。
- 搜索源码指纹、协议、预算、checkpoint_interval 均不变；GT 短暂停顿仍计入原墙钟预算。运行时长受存储调整影响，不能直接将前后不同长度样本的耗时相除作为加速比。

### 用户要求停止（2026-10-23）

- 用户需要修复 NAS，已明确要求停止；四个运行进程全部退出，GPU 1、2 显存归零，不自动重启。
- 捕获 165/165；GT 完成 32/165、Random 完成 12/165。两份在途检查点有效，已最后同步到 NAS，44 份完成结果均已镜像。
- 暂停详情：新产物目录 `setup/io-fix/stopped-by-user.json`。等待用户明确要求继续后才准备检查点续跑；不重跑已完成任务，不改变原预算。

### NAS 修复后的显式续跑（2026-10-23）

- 用户告知 NAS 已修复，检查模型、冻结源码、44 份完成结果及两个暂停检查点均通过；GPU 1、2 空闲。
- 复用原冻结攻击的恢复逻辑，仅对 GT `fedsgd-b2-02` 和 Random `fedsgd-b1-12` 开启 resume；不重跑完整结果，不清零时间或评分次数，暂停等待时间不计入预算。
- controller 6732、watcher 6749；GT worker 6737（GPU 2）、Random worker 6740（GPU 1）。续跑脚本与审计在新产物目录 `setup/resume-1/`，原 controller 不可重复启动。
- 继续本地临时盘写入和 NAS 异步镜像；每十分钟巡检。原有续跑测试 5 项和包装入口一致性测试 1 项通过；恢复后的首次检查点、累计预算与原结果不变性均已核验通过，详见 `setup/resume-1/verified.json`。

### 完成收尾（2026-10-25）

- 330 次攻击及统一评分、报告已于 2026-10-24 02:39 完成；全部 completed，无预算耗尽。配对上传、源码与暂停前 44 份结果哈希通过，报告九个文件 SHA256 通过。
- 全部 worker、controller、watcher 已退出，GPU 1、2 显存为零；所有攻击产物已归档 NAS，标准路径不再依赖远端临时缓存。
- 实验页以完整 30 样本为当前结果，pilot 留作前期审计。完整表图在新目录 `report/`，文档副本及来源清单位于 `docs/experiments/data/federated-text-pilot/slake30-manifest.json`。没有提交 Git，也没有启动额外实验。

## 单步 SGD 精度对照（2026-10-25）

用户授权验证数值路径与参数精度对恢复的独立影响。本阶段复用三十样本的十五组公开 batch=2 梯度，不重新采样，不重新计算梯度；暂不测多步。

| Item | Setting |
|---|---|
| Cohort | 原 SLAKE 三十条问答、十五组 batch=2 配对，原 LLaVA rank 32 Round 10 |
| Input gradient | 原 fedsgd-b2 上传，bf16；全部新条件使用逐字节相同的 G |
| Ideal control | 用 float64 构造 -lr×G，不执行参数相减，检查标量缩放等价性 |
| Standard bf16 | 实际 torch.optim.SGD，foreach=False、fused=False；bf16 参数更新，再取 bf16 差值 |
| Standard FP32 | 同一初始参数和同一梯度转 FP32，实际 torch.optim.SGD 更新并保留 FP32 差值；不是 FP32 前后向训练 |
| Legacy reference | 原 fedavg-b2-s1 的 0/30 结果保留；原 gradient 基线为 GT 10/30、Random 6/30，不覆盖旧产物 |
| Correction | core.fl 的函数式 SGD 使用 torch.add(param, gradient, alpha=-lr)，对齐标准优化器且保留可微回放 |
| Attack | 原 token filter、beam=16、LM prior=0.01、rerank=0；每样本 200000 evaluations 和 3600 秒 |
| Matrix | 三新条件 × 十五上传 × GT/Random，共九十次新攻击 |
| Stages | 准备公开上传，先完成全部 token filter 并保存检查点，再续跑序列；统一评分前所有重构必须保存 |
| Budget | token filter 耗时计入原预算；等待后续阶段的空闲时间不计入，计数不重置 |
| Evaluation | 保留总体 n=30、分桶 n=10，一对一文本匹配；评分阶段才读取原私有参考 |
| Output | outputs/diagnostics/sgd_precision_slake30_20261025/，代码冻结为独立 source-v1 |
| Runtime | 先检查 WN-A100 空闲卡，最多两张；GPU 1 目前被其他任务占用，准备使用空闲 GPU 2、5；GPU 0 随后被占用，未启动攻击即更换预选卡 |

- [x] SGD 数值修正；bf16、FP32、FP64 单步与多步对照标准优化器，保留输入高阶梯度。
- [x] 固定梯度实验与延迟评分入口；定向测试 114 通过、2 跳过，Ruff 通过。
- [x] 全量 CPU 回归 378 通过、4 跳过；独立源码已冻结。45 份上传及数值审计准备完成。
- [x] GPU bf16 与 FP32 算子对照实际 SGD 逐坐标一致，首份过滤检查点已保存。
- [x] 完成 token filter、九十次攻击与统一评分；全部 completed，无预算耗尽和 OOM。
- [x] 精度对照结果已追加到 [[federated-text-pilot]]，原实现、标准 bf16、FP32 结果及范围均明确；没有提交 Git。

单步对照已启动：初始 controller 6848、watcher 6853 在 GPU 0 被占用后退出；新 controller 6877、watcher 6881 接管原 CPU 准备 worker 6851，没有重跑准备。当前尚在准备公开上传，之后只在空闲 GPU 2、5 上最多双卡运行。固定源码 `a2c3e9d3f0d4e5907a78510832f320ae4312e2e7ae7667ca1a40b3200d0cfdd5`，基础 commit 仍为 `31ef506`，本轮不提交。

准备阶段已完成，GT worker 6892（GPU 2）、Random worker 6895（GPU 5）已实际运行；CUDA SGD 一致性检查两边均 passed。全部十五组的所检 LoRA-A 块平均非零比例：旧实现 6.71%、标准 bf16 6.73%、标准 FP32 99.84%；这是数值审计，不是重构率。

用户关注显存风险后，新增纯 nvidia-smi 三十秒巡检 PID 7034，不创建 CUDA 上下文；两卡只运行各自 worker，暂未 OOM。全部九十份过滤检查点已完成，正在执行序列。当前卡接近满显存，额外 CPU/CUDA 上传算术比对需等 worker 退出后再执行；不得因风险自动改变 batch、dtype 或预算。

### 精度对照收尾（2026-10-25）

- 九十次新攻击及评分、表图于 18:41 完成，全部 completed，无 OOM 和预算耗尽；所有任务及两种巡检都已退出。
- 理想差值与 FP32 更新的 Joint EM 均恢复原梯度基线，标准 bf16 仍归零；本轮主要损失来自参数更新精度，不能据原两步零恢复单独估计步数影响。
- 更新算术在 CPU 标准 PyTorch SGD 执行，攻击在 GPU 执行；FP32 没有重新前后向。任务结束后在空闲 GPU 2 完成所检十五组的 CPU/CUDA 更新对照：FP32 完全一致，bf16 99.9473% 一致；没有另做 CUDA bf16 差值的完整攻击。
- 原结果及源码不覆盖。完整比较表图与范围说明在 `report/final-comparison.json`，文档副本清单为 `docs/experiments/data/federated-text-pilot/precision-manifest.json`。本轮没有提交 Git，也未启动新的多步或训练实验。

## 真实 FP32 多步更新（2026-10-25）

用户授权在每一步重新前后向计算梯度，使用 FP32 执行多步本地更新。本阶段与“固定同一份旧梯度”的算术对照分开记录，不覆盖原结果。

| Item | Setting |
|---|---|
| Model | 原 SLAKE LLaVA-1.5-7B rank 32、alpha 64、Round 10；先恢复并核验原 bf16 权重，再将全部原数值精确提升至 FP32 |
| Precision | 全模型前向、反向、SGD 参数更新和上传均为 FP32；关闭 autocast、matmul TF32 和 cuDNN TF32 |
| Initial values | 不直接以 FP32 重载原始预训练权重，避免悄悄换成不同的冻结基座数值；capture 与 attacker 采用同一公开转换规则和派生指纹 |
| Dataset | 原三十个冻结 SLAKE QA，原顺序相邻配对，三个长度桶各十条，同一客户端 |
| Conditions | FP32 FedSGD batch=2；FP32 FedAvg batch=2、steps=1；FP32 FedAvg batch=1、steps=2 |
| Sample count | 每份上传固定两条不同问答；两步各访问一条，单次访问，无重复 local epochs |
| Optimizer | 标准 SGD 数值路径，lr=0.01，无 momentum、weight decay 或梯度累积 |
| Trace | 记录实际前向次数及所检 LoRA-A 块距离初始参数的范数；第二次前向必须使用第一步更新后的参数 |
| Control audit | 单步参数差值与同一配对重新计算的 FP32 梯度比较；保留函数式回放与实际 torch.optim.SGD 测试 |
| Attack | DAGER-Erasure-LM，K=100、beam=16、LM prior=0.01、rerank=0；GT 与 Random 配对 |
| Matrix | 三条件、四十五份上传、九十次新攻击；总体 n=30，分桶 n=10 |
| Budget | 每样本 200000 evaluations、3600 秒，双样本上传乘二；异常和 OOM 不自动重试，不静默改变参数 |
| Boundaries | B2/S1 与 B1/S2 也改变 minibatch loss 加权方式；不能把所有差异归因于模型漂移或步数；原 bf16 与本轮新梯度之间还改变了前后向精度 |
| Resources | WN-A100 最多两张空闲 A100；预选 GPU 2、5，启动前再次核验；沿用共享 gia 环境 |
| Output | outputs/diagnostics/fp32_multistep_slake30_20261025/ |

- [x] 新增独立全 FP32 capture、逐步参数轨迹与攻击入口，保持原模型文件不变。
- [x] 真实双步重放对照实际 PyTorch SGD、全参数精度与初始数值保持测试；定向测试 94 通过、2 跳过，Ruff 通过。
- [x] 全量 CPU 回归 381 通过、4 跳过；固定源码，启动 detached controller 7651 与 watcher 7657、显存巡检 7659。
- [x] 公开双步 smoke 与首份真实 FP32 捕获均 passed；448 个 F32 上传张量，peak allocated 约 45.0 GiB，第二次前向参数变化已核验。
- [ ] 本阶段由用户要求停止，九十次完整矩阵不再继续；不补零，不提交 Git。

真实 FP32 阶段已在 WN-A100 启动，源码指纹 `33f41d5766f96e0e720090ffc976d8e41179296fcd466219ae87f4e405c3ca21`；当前先做公开双步 smoke 与捕获，再启动两种图像攻击。首份实际捕获审计待完成，不提前报告恢复率。


### 全 FP32 停止并切换混合精度（2026-10-25）

用户确认切换方案。全 FP32 已停止，45/45 捕获保留，只完成 2/90 攻击；另两份在途检查点已保存 NAS。旧五个进程退出、GPU 2/5 释放，停止审计在旧目录 `setup/stop-audit.json`。不把未完成矩阵当作恢复率结论。

## 混合精度真实多步更新（2026-10-25）

- 冻结骨干维持原 bf16 数值；可训练 LoRA 提升为 FP32，在 bf16 autocast 下计算每步新梯度，SGD 与最终上传差值为 FP32。
- autocast cache 关闭；每个本地步骤都用已更新的 FP32 LoRA 参数重新前后向，不复用固定旧梯度。
- 原数据、初始 Round-10 模型、三条件、配对、lr、候选与搜索预算保持一致；新矩阵仍为 45 捕获、90 次配对攻击。
- capture 和 attacker 均用同一混合精度转换与 autocast；结果明确记录不是全 FP32，不混用旧捕获或重构。
- 新目录 `outputs/diagnostics/mixed_multistep_slake30_20261025/`，沿用 gia；本地编辑并冻结源码，WN-A100 空闲 GPU 2/5 最多双卡；不提交 Git。
- 异常或 OOM 不自动重试或修改参数；首次合成 smoke 和真实捕获检查通过后再进入攻击；全部结果保存后统一评分。

- [x] 已停止全 FP32 并归档两份在途检查点与完整结果。
- [x] 混合精度双步 CPU 对照实际 torch.optim.SGD、初值保持及 FP32 更新未被 bf16 舍入测试通过；定向 24 项通过。
- [x] 全量 CPU 回归 382 通过、4 跳过，Ruff 通过；源码 `529566c53e748be0941f0720c73b15a112e232f09f62be6becfbcebf748e28e3` 已冻结，detached 启动与监控已提交。
- [x] 首份 GPU 双步 smoke 和实际捕获 passed：448 个 F32 张量，bf16 投影输出、第二步更新后参数；首捕获峰值约 29.0 GiB。
- [x] 首个攻击检查点、全部 45 份捕获与 90 次攻击已核验；十五份双步轨迹确认每步使用更新后参数。
- [x] 90 次攻击全部 completed、统一评分与聚合表图已生成，既有实验页、track 和首页结论已同步；不提交 Git。

混合精度任务于 2026-10-25 23:19 完成，所有 worker、controller 和监控均已退出，GPU 2/5 已释放。45 份捕获和 90 次攻击全部完成，统一评分与表图见新目录 report/；全部结果已归档 NAS。首份 smoke、捕获、十五份双步轨迹、配对上传和源码精度核验通过。单步恢复 GT 10/30、Random 6/30；双步仍恢复 9/30、5/30，结论与限制见 [[federated-text-pilot]]。


## AdamW 单步舍入小验证（2026-10-25）

用户授权用 FedVLMBench 的 AdamW 初始 lr=1e-4 检查更新幅度与舍入。复用原十五组公开 bf16 batch=2 梯度及原 Round-10 六个选定 LoRA-A 权重，CPU 实际 torch.optim 单步算术，不占新 GPU、不重算梯度、不启动文本重构。固定零初始动量、betas=(0.9,0.999)、eps=1e-8、weight_decay=0（与 encoder-based 生成配置一致），对照 SGD lr=0.01 与 AdamW lr=1e-4，各测 bf16 和 FP32 参数/状态，并用 FP64 理想更新只舍入到 bf16 参数的路径分离误差。

- [x] 公开梯度与权重哈希核验，原混合精度后台实验继续运行，未改变任何设置。
- [x] 首步公式与实际 FP64 AdamW 相对误差 4.17e-15；SGD bf16 零差值比例 93.2743% 精确复现。
- [x] 聚合 JSON、CSV、来源哈希已保存，结果已追加既有实验页；AdamW bf16 零差值约 0.01646%，幅值相对 L2 误差约 16.49%。没有重构评分、额外 GPU 或 Git 提交。

输出 `outputs/diagnostics/adamw_rounding_slake30_20261025/`；脚本 `setup/run.py`，沿用 gia Python 在本机执行 CPU 算术。只读取 public 上传和公开模型，不读取图像、文本真值或重构结果。

## 待做：FedVLMBench AdamW 多步实验 {#fedvlmbench-adamw}

用户于 2026-10-25 要求登记并提醒。用户随后已明确要求执行，四样本验证已于 2026-10-26 00:24 完成。现有 AdamW 单步舍入审计已完成，本项针对尚未开展的多步文本重构，二者分开记录。

| Item | Planned setting |
|---|---|
| Goal | 检查更贴近 FedVLMBench 的 AdamW 本地更新下，单步与多步文本泄露如何变化 |
| Optimizer | AdamW，初始 lr=1e-4，betas=(0.9,0.999)、eps=1e-8、weight_decay=0；执行前核对目标配置及状态重置规则 |
| Main precision | bf16 前后向；FP32 LoRA 参数、梯度累积、optimizer state 和参数更新；上传 FP32 参数差值 |
| Control | 同一 AdamW 设置下对照纯 bf16 参数及状态，分离参数量化影响；已有 SGD 对照继续保留 |
| Data and model | 优先复用冻结 SLAKE 队列与 LLaVA rank 32、Round 10；先小规模验证，再确定正式矩阵 |
| Local steps | 包含单步基线与多步；每一步在当前参数上重新计算梯度并更新动量，固定数据访问及 loss 归一化口径 |
| Reconstruction | GT/Random 配对，公开观察与私有评分分离；不把数值误差或 unsupported 状态当作重构结果 |
| Prerequisite | 当前 pooled DAGER 只支持 plain SGD；先分析 AdamW 逐坐标预条件对梯度子空间关系的影响，验证方法适配，不能仅移除支持检查 |
| Fidelity | lr=1e-4 是已核对的初始配置；完整训练调度、精度与优化器状态策略需另行核实，局部更新替换不称为完整复现 |

- [x] 现有 SGD 混合精度结果已完成；本次汇总时提醒用户跟进 AdamW 待做项。
- [x] 五条件、四个固定 QA、十份上传、二十次 GT/Random 攻击；WN-A100 GPU 2/5，最多两卡。
- [x] 生成器初始 lr=1e-4、weight_decay=0，默认 adamw_torch；训练入口每轮每客户端新建 Trainer，按 round 余弦衰减到 1e-6，训练参数构造器显式 fp16=True。本轮借鉴优化器与初始 lr，采用所约定 bf16 计算，不称为完整复现；optimizer state 每份上传从零开始，内部两步保留动量。
- [x] 数学反例验证 AdamW 不保留原梯度行空间；默认继续拒绝 AdamW，新增显式差值提案诊断，必须使用实际优化器全上传重放重排。候选阶段仍是启发式，不声称精确梯度恢复。
- [x] FP32 LoRA/状态、bf16 计算的单步及双步验证已完成，均为 GT/Random Joint EM 0/4。
- [x] 纯 bf16 参数及状态对照完成，单步、双步 Joint EM 均 0/4；现阶段不扩大正式矩阵，转向候选生成问题。

混合精度收尾检查：90 次状态均 completed，180 个条件内 QA 行及六组总体指标重算通过，九个报告哈希与十二份文档聚合副本哈希一致；原始实验与全 FP32 中止产物不覆盖，无新 GPU 实验或 Git 提交。


### AdamW 四样本验证执行（2026-10-25）

| Item | Setting |
|---|---|
| Cohort | 复用原 select_pilot：原冻结队列中两条短、两条中等长度 QA，固定为两个上传组；按规则预先选取，不按恢复结果选择 |
| Conditions | Mixed gradient control B2/S1；AdamW FP32 master B2/S1、B1/S2；AdamW bf16 weights/state B2/S1、B1/S2 |
| Precision | 所有条件 bf16 autocast，关闭 cast cache 与 TF32；只改变 LoRA 参数、梯度及 optimizer state 的 FP32/bf16 存储 |
| Optimizer | AdamW lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0；每步新梯度，上传内动量持续；梯度控制为 SGD/FedSGD |
| Proposal | K=100、beam=16、LM prior=0.01，AdamW delta 子空间仅用于启发式提案；不把它标作真实梯度子空间 |
| Selection | 五条件都启用 rerank_candidates=16；按声明的优化器重放完整上传，以观测差值的全局相对平方 L2 选取；Random 使用替代图，存在模态失配 |
| Fidelity | core.fl 的 AdamW 算术对齐标准单张量 torch.optim.AdamW：lerp、addcmul、addcdiv；不是简单放开支持检查 |
| Tests | 定向 97 通过、2 跳过；新增研究测试 12 通过；最终全量 CPU 394 通过、4 跳过，Ruff 通过 |
| Source | outputs/diagnostics/adamw_text_pilot_20261025/source-v1，指纹 3c7cbf286c6f4e8b334069254e3da10df802a57d18c11e9c02959fa9d3d264b5 |
| Runtime | 本地 gia 环境未改，WN-A100 空闲 GPU 2/5 最多双卡；detached controller、十分钟状态巡检及三十秒显存巡检 |
| Boundaries | 四个 QA 只能作 pilot；B2/S1 对 B1/S2 改变 batch/loss 加权；未重新训练 Round-10 模型或按轮执行学习率衰减；没有保证第一阶段包含真实 token |

- [x] 原有结果、paper 与文档保留；新增代码及本轮结果暂不提交 Git。
- [x] 完成 CPU 验证、冻结源码并启动远程任务。
- [x] FP32 与 bf16 的 GPU 原生双步 smoke 均与实际 AdamW 逐项完全一致；捕获、攻击检查点与所有双步轨迹通过。
- [x] 十份捕获、二十次重构及评分完成，全部 completed；共 320 次优化器重放，私有参考只在重构保存后评分。
- [x] 表图及方法局限已追加既有实验页，十份聚合副本与来源哈希一致；当前不扩三十样本矩阵。

本阶段远程 PID：controller 9199、capture worker 9204（GPU 2）、watcher 9206、显存巡检 9208。启动时 GPU 2/5 均空闲；模型正在从 NAS 加载，首份 GPU smoke 与捕获待核验；实际状态以新目录 controller.json 和远端进程为准。

首次 GPU 核验：FP32 LoRA/状态的双步回放与实际 AdamW 逐项完全一致，公开 smoke passed。首份真实上传审计 passed：448 个 F32 张量，GT/Random 哈希一致，Random 无公开图像；当前捕获 3/10，BF16 对照 smoke 与攻击检查点待核验。详情在新目录 public-smoke-mixed.json、setup/first-capture-audit.json。本次未提交 Git，状态巡检与显存巡检持续运行。


### AdamW pilot 完成与下一步判断（2026-10-26）

- 00:24 完成，20/20 completed，无 OOM 或预算耗尽，全部后台任务及监控退出，GPU 2/5 释放；结果和表图见 [[federated-text-pilot]]，收尾审计在新目录 `setup/finalization-audit.json`。
- AdamW FP32/bf16、单步/双步均为 0/4；FP32 单步候选召回 GT 0.577、Random 0.612，完整覆盖 0/4；双步覆盖 1/4，搜索仍无完整正确候选。既有十六个候选的事后最优上界全部为 0/4，不是最终重排漏选正确问答。
- 公开上传的直接解析反变换诊断：单步 FP32 六个 LoRA-A 块，有限坐标的梯度相对误差约 80%，误差放大中位数约 1838。只套闭式公式不足以作为稳定的攻击适配。
- 决策：暂不扩大到三十 QA，不继续堆叠 beam/rerank 预算；下一研究重点为单步 FP32 AdamW 的候选生成及梯度子空间可恢复性。先接近同梯度基线候选覆盖，再讨论序列搜索、多步和独立样本验证。
- [ ] 若继续，先确定一种 AdamW-aware token filter 的小规模验证方法；不以本次四个 QA 的零恢复声称隐私保证。
- 本轮新增代码、文档及结果均未提交；paper 保留，不启动新的 GPU 实验。
