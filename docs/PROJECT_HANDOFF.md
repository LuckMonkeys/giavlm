# 项目交接

## 2026-10-28 文本实验与站点提交

- 按用户要求，既有文本实验代码、记录与网页站点修改提交到 main；今天的图像分析使用独立分支。代码提交 270eb81 包含 AdamW 回放数值适配、显式差值提案和固定 QA 数量的 SGD 对照，原研究边界不变。
- 文本结果见 [[federated-text-pilot]]；站点新增 KaTeX 数学渲染和一级 Plans 入口。原始运行产物留在 outputs；论文草稿不纳入本次提交，也不推送远端。
- 分支隔离副本中代码回归四百项通过、四项跳过；索引重建后全部三十六项文档测试通过，Ruff 通过。图像工作未混入 main 的实验目录与导航。

## 2026-10-26 文本恢复优先研究防御鲁棒性

- 用户明确将防御条件下的文本恢复退化列为当前优先问题；先研究加噪与稀疏化下的候选漏检，再考虑序列搜索优化。全局裁剪保留作对照，既有结果见 [[federated-text-pilot]]。
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
