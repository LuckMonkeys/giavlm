# 跨模型、跨数据集与长度的文本恢复实验计划

- 更新时间：2026-10-18（Asia/Shanghai）。
- 状态：已完成。六个组合的十轮训练与 288 次攻击全部完成；完整表图、ZIP、实验记录及文档检查均通过；代码已提交为 2440189、6beafa7。
- 基线代码：分支 `dager-rank-and-experiment-records`，起点 commit `4d11308`。后续运行必须另记实际源码指纹和未提交差异。
- 本文件保存执行计划与检查清单；阶段结果写入 `docs/experiments/`，当前任务与进程写入 `docs/PROJECT_HANDOFF.md`。不能把计划值当作测量结果。

## 已确认的目标与边界

- 测试 LLaVA-1.5-7B 与 Qwen3.5-4B 的文本恢复效果。
- 数据覆盖 SLAKE、VQA-RAD、VQAv2，比较不同文本长度。
- 使用最新 DAGER-Erasure-LM：`dager_adapted` 的 `public_residual` 筛选与纯文本 LM prior，`lm_prior_weight=0.01`。
- 主实验使用 LoRA rank 32、alpha 64，各模型在各数据集上训练到 Round 10 后再捕获攻击观测。这里的 10 轮指 federated rounds，不是遍历数据集 10 个 epochs。
- 每个样本比较 Ground-Truth 图像与 Random 替代图像；两种知识条件使用同一个受害模型上传更新。
- 代码修改、数据准备、Conda 环境创建和依赖安装全部在本机完成。文件保存在 NAS，WN-A100 直接使用。
- GPU 程序经 SSH 在 WN-A100 的空闲卡上运行。本项目全部任务同时最多占用两张物理 GPU，训练、攻击与 GPU smoke test 合并计数。
- 保留原 `gia` 环境。新改动先保存在工作区，本轮不自动提交 Git。

## 实验矩阵与固定条件

| Item | Setting |
|---|---|
| Model | LLaVA-1.5-7B; Qwen3.5-4B |
| Dataset | SLAKE; VQA-RAD; VQAv2 |
| Training | F-L; FedAvg; AdamW; lr 0.0001 |
| LoRA | rank 32; alpha 64; dropout 0 |
| Training budget | 10 rounds; 1 client per round; 10 local optimizer steps per round |
| Batch | batch size 2; gradient accumulation 4 |
| Seeds | model initialization 17; training and data split 42 |
| Model checkpoint | Round 0, Round 5, Round 10; 主表使用 Round 10 |
| Capture | 单样本、单步 FedSGD；从冻结 Round 10 模型另行捕获 |
| Attack | DAGER-Erasure-LM; LM prior weight 0.01 |
| Image | Ground-Truth (`image_known`); Random (`private`) |
| Public information | 精确内容 token 长度显式公开；GT 条件另外公开图像 |
| Length bins | 内容 Q+A 的参考 token 数：1–16, 17–32, 33–64 |
| Planned sample count | 每个数据集每桶 10 个不同图像的 QA；不足则报告实际数量 |
| Pilot | SLAKE; 2 models × 3 bins × 10 samples × 2 image conditions，最多 120 次攻击 |
| Full benchmark | 3 datasets × 2 models × 3 bins × 10 samples × 2 image conditions，最多 360 次攻击，包含 pilot |

训练预算为每个模型与数据集组合 100 次本地 optimizer steps、800 次样本呈现，不代表 800 个不同样本。每个组合独立从该模型的公开基础权重起步；同一模型的各数据集训练共享相同初始 LoRA 基础。训练用 AdamW，攻击捕获用 FedSGD，两者分别记录。

参考长度使用固定的 LLaVA tokenizer，排除公开模板、视觉 token、EOS 和 padding。跨模型使用同一批原始 QA，另报每个模型原生 tokenizer 下的 Q、A 长度。33–64 是总内容长度，不保证每段都能放进当前槽位；必须先审计，再冻结容量，禁止静默截断。

## 分阶段检查清单

### Phase 0：环境与数据准备

- [x] 核查工作区、已有提交和已有实验状态。
- [x] 验证 SSH 别名 `zx-wn-a100`、NAS 仓库、原环境与 Qwen 权重在远端可读。
- [x] 保存 `gia` 的 pip 与 Conda 依赖快照。
- [x] 在本机以独立文件副本创建 NAS 环境 `gia-qwen35`，安装支持 Qwen3.5 的固定版本依赖。
- [x] 保存新环境的版本清单，确认旧 `gia` 未发生变更。
- [x] 在新环境中核对实际 Qwen3.5 模块接口、LoRA 注入位置与梯度计算支持。
- [x] 准备 VQA-RAD、VQAv2 标准 manifest，并审计现有 SLAKE。
- [x] 审计图像分组、训练与评估隔离、重复 QA、长度分布、原生 token 长度及容量溢出。
- [x] 冻结样本清单、抽样 seed、tokenizer 版本、模型 revision、数据指纹与输入容量。

数据清单与逐样本审计保存在忽略的 `outputs/` 下，公开文档只保留聚合信息。优先为长桶分配图像，保证最终各桶图像互不重复；抽样依据数据与长度，不依据恢复结果。若数据源、可用样本量或下载方式迫使采用子集，必须明确记录子集定义，不能称为全量数据训练。

### Phase 1：Qwen3.5 适配与正确性验证

- [x] 增加独立模型适配器与配置注册，旧环境不提前导入仅新 Transformers 提供的类。
- [x] 保持受害模型输入模板、视觉 token、位置编码、监督 mask 与攻击视图一致。
- [x] 检验首层及第二层的 LoRA 输入梯度与特征子空间关系。
- [x] 检验部分前向得到的层输入与完整前向 hook 一致。
- [x] 检验纯文本 LM prior 的下一 token 对齐，不读入真实文本或未来 token。
- [x] 通过有针对性的 CPU 测试和原 LLaVA 回归。
- [x] 在 WN-A100 一张空闲卡上验证真实权重的前向、反向、捕获、保存和单样本攻击。

Qwen3.5 前部包含 linear attention，不能直接沿用 LLaVA 的前两层 q/k/v 模块路径。必须从实际架构验证可观测投影及数学关系，并报告观测矩阵的尺寸与秩；相同 LoRA rank 不自动表示两模型可用约束数相同。若存在不支持环节，应修复或明确报告，不能将未实现记为重构失败。

### Phase 2：十轮受害模型训练

- [x] 对照旧 LLaVA/SLAKE rank 32 快照的训练协议、数据、输入容量与模型指纹。
- [x] 仅在全部一致时复用旧快照；否则重训，共最多 6 个模型与数据集组合。
- [x] 每个任务冻结启动时源码、配置、环境和数据指纹。
- [x] 保存 Round 0、5、10，以及断点恢复所需状态。
- [x] 检查首个有效 checkpoint、训练损失及最终 Round 10 完成状态。

已核验：SLAKE 原始答案最长 44 个 LLaVA token，旧模型 A32 会截断；本轮统一采用 Q64、A64，旧快照不复用，六个模型与数据集组合均从头训练。不能在加载旧快照后把改变容量或数据的实验称为相同训练条件。

### Phase 3：SLAKE pilot

- [x] 在 Round 10 上为冻结样本捕获单样本 FedSGD 更新，不修改全局模型。
- [x] 配对运行 GT 与 Random，固定随机图像 seed，并记录替代图像的生成规则。
- [x] 验证第一个 capture 与 reconstruction 产物及评估边界。
- [x] 使用相同 filter、beam 和预算策略，不根据测试真值选择候选或调参。
- [x] 报告 token 筛选、完整序列恢复、耗时、显存与失败状态。
- [x] 完成 SLAKE 阶段记录，再扩展其余数据集。

起始搜索条件沿用现有设置：informative token top-k 100、beam 16、prefix batch 32、最多 200000 次评估、3600 秒。正式启动前核对配置中的准确键名，记录必需的 ambiguous token 加入规则。两模型采用相同已声明规则；如正确性或预算审计要求修改，应在正式结果前冻结并记录，不能失败后静默降配或对个别样本放宽。

### Phase 4：VQA-RAD 与 VQAv2

- [x] 复用已验证的训练与攻击流程，运行对应 Round 10 快照。
- [x] 保持已冻结长度分桶、样本、超参数和预算，分别报告原生长度。
- [x] 核验每个计划任务都有成功、失败或不支持的终态，没有遗漏与伪完成。

### Phase 5：汇总与交付

- [x] 汇总主表与分长度表，并记录每行实际样本数。
- [x] 生成模型、长度与图像条件的聚合对照图。
- [x] 保存小型、不含私有文本与样本标识的聚合 JSON，并核对来源指纹。
- [x] 完成实验记录、实验索引与交接更新，运行相关验证。
- [x] 检查没有遗留训练、攻击或 watcher 进程。

## 报告格式与完成标准

主表按 `Model | Dataset | Image | N | Joint EM | ROUGE-1 | ROUGE-2 | ROUGE-L | Word Recall` 展示；分长度表增加 `Length bin`。`Joint EM` 指问题和答案同时精确恢复，沿用项目评估规范，另报 Question EM 与 Answer EM。

token filter 表按 `Model | Dataset | Length bin | Image | N | Recall@100 | AP | Full Coverage` 展示。明确 Recall@100 的候选口径，以及 Full Coverage 是否计入额外的 ambiguous token。另保存预算耗尽、运行错误、不支持、耗时和峰值显存，避免只展示成功样本。

图至少包含按长度的 Joint EM 曲线或柱状图，以及 GT 与 Random 的恢复差距。数据集与模型分组展示，不能把不同文本长度的总体平均值直接解释为架构优势。

完成条件：所有冻结任务获得可核验终态，表图与实验记录生成，相关检查通过。无法运行的组合需要给出具体原因与验证证据；队列启动、训练结束或单个样本成功均不代表整个 benchmark 完成。

## 运行、故障与恢复规则

- 本机环境路径：`/home/zx/nas/miniconda3/envs/gia-qwen35`。LLaVA 优先继续使用旧 `gia`，Qwen 使用新环境，各自固定版本。
- GPU 执行主机：`zx-wn-a100`。启动前重新检查远端 GPU、进程与项目已有任务；使用远端 GPU UUID 绑定设备。
- 同时最多两张物理卡，优先一个任务一张卡。此前 GPU 4、5 空闲只是历史观察，不是预约。
- 长任务在远端 detached 启动，记录 host、GPU UUID、环境、源码指纹、命令、PID、开始时间、日志和输出目录，并配置完成、失败、主进程退出时终止的 watcher。
- 不终止其他任务；不 OOM 自动重试；不自动改变 batch、dtype、更新模式、本地步数或知识条件。异常保存失败状态并抛出。
- 运行时保持源码不变，或从隔离快照执行；共享 NAS 上的本地改动会立即影响远端新加载的代码。
- 不向攻击器传入私有图像、文本或标识；只有声明公开的字段通过 Observation 进入攻击器。重构提交后才由 evaluation 读真值评分。
- 中断后先核对实际 PID、输出与 checkpoint，不能只看调度表的 `running` 状态。只有协议和指纹一致的任务才可续跑。

## 当前执行记录

| Time | Phase | State | Evidence / Next action |
|---|---|---|---|
| 2026-10-17 | Preflight | Done | 起点 `4d11308`；SSH、NAS 与两张空闲 A100 已核验 |
| 2026-10-17 | Environment | Done | 新环境 Transformers 5.20.1、PEFT 0.21.2、Accelerate 1.15.0；Torch 2.6.0；旧环境 pip 快照一致 |
| 2026-10-17 | Adapter gate | Done | 新环境 Qwen 离线检查 3 通过；旧环境完整回归 335 通过、4 跳过；真实权重 smoke 通过 |
| 2026-10-17 | Frozen data | Done | 三数据集 72 个 QA；完整抽样池不变，VQAv2 固定调度的 816 张用图校验通过 |
| 2026-10-18 | Training | Done | 六个组合均完成 Round 10，保存 Round 0、5、10；共享初始 LoRA 与文件哈希通过核验 |
| 2026-10-18 | SLAKE | Done | 两模型共 120 次攻击，均为 completed；统一落盘后评估 |
| 2026-10-18 | VQA-RAD | Done | 两模型共 84 次攻击，均为 completed；统一落盘后评估 |
| 2026-10-18 | VQAv2 | Done | 两模型共 84 次攻击，均为 completed；统一落盘后评估 |
| 2026-10-18 | Final report | Done | 6/6 组合，288/288 completed，18 文件 ZIP CRC 与报告来源 SHA256 校验通过 |
| 2026-10-18 | Scheduled monitor | Done | 每 600 秒巡检，共 31 次，最终 completed 后退出；会话主动巡检也持续至完成 |
| 2026-10-18 | Cleanup verification | Done | 远端无遗留本轮进程，本机汇总与巡检均退出；GPU 4、5 显存均为 0 MiB |
| 2026-10-18 | Documentation | Done | 17 份实验记录与产物路径检查通过；MkDocs strict build 通过；新代码和聚合结果当时未提交 |

正式输入容量已固定为 Q64、A64；初步审计显示三数据集均可容纳。VQA-RAD 和 VQAv2 的长桶各只有一张评估图像，正式数量以冻结清单为准。VQAv2 保留全部官方 train/val 标注作为抽样池，只下载固定训练调度与评估样本用到的图像；这不改变训练采样分布。

最终矩阵为 SLAKE 30 个 QA、VQA-RAD 21 个 QA、VQAv2 21 个 QA，合计 72 个 QA；两个模型、两个图像条件共 288 次攻击。SLAKE pilot 仍为 120 次。原先的 360 是每桶都能取满时的上限。各模型原生长度及训练池 QA 重叠率见 `setup/selection-audit.json`；图像隔离不代表文本内容从未出现在训练池中。

正式攻击统一将 vocabulary chunk 从 1024 调整为 8192，以减少大词表的检查点开销；top-k、beam、prefix batch、评分和预算不变。该配置在正式攻击前冻结。训练和攻击使用不同的源码快照，均保留逐文件哈希清单。

本轮已完成。接手时先读本轮根目录的 `controller.json` 与 `finalizer.json`，再检查 `jobs/*.json`；不要根据历史 PID 启动重复任务。不要重复启动调度器或已经存在的任务。控制器失败时不会重试 GPU 任务；保留失败目录，先诊断原因。汇总目录为 `report/`，自动 ZIP 只在六个组合完成后生成；在此之前，图中可能显示 Pending，不能当作最终结果。

## 定期巡检

- 用户于 2026-10-17 要求增加定时运行状态检查；已部署独立后台任务，每 10 分钟检查一次，启动后立即执行首次检查。
- 当前运行环境无 `crontab`，使用 detached CPU 定时进程，与当前对话和 SSH 连接断开无关；主机重启后需重新启动。
- 检查远端调度器、worker 与子进程的实际 PID 和命令，查询 GPU，占用上限仍为两张。
- 汇总训练轮次、观测捕获数量、已结束攻击数量、当前 checkpoint 的 stage/depth，以及本机汇总进程是否存活。
- SSH 探测失败时标记未核验，进度停滞和失败写入异常记录；不重启任务、不改变配置、不启动额外 GPU 作业。
- 在调度器失败、汇总任务失败或最终报告生成完成后结束巡检。
- 最新报告：`monitor/latest.md`、`monitor/latest.json`；历史：`monitor/history.jsonl`；最近异常：`monitor/alert.json`（发生异常时生成）；定时器状态与下次检查时间：`monitor/timer.json`。
- 正式 600 秒任务已执行 31 次检查并正常退出，原 PID 1525580；最终状态 completed，无下一次检查。
- 入口为 `python -m utils.experiment_monitor --root outputs/diagnostics/crossmodel_text_20261017 --interval 600`；部署脚本为 `setup/launch-status-monitor.sh`。重启前先检查旧 PID 和 `monitor.lock`，避免重复运行。

上述文件路径均相对于本轮产物根目录 `outputs/diagnostics/crossmodel_text_20261017/`。巡检结果保存在文件中，不自动向聊天、邮件或其他应用发送消息。已结束攻击数量不代表精确恢复成功数量。

本轮产物根目录为 `outputs/diagnostics/crossmodel_text_20261017/`。环境准备日志为 `setup/conda-clone.log`，旧环境快照为 `setup/gia-before-pip.txt` 与 `setup/gia-before-conda.txt`。后续每阶段完成、失败或范围变化时更新检查清单与本表；不得在未检查产物时提前勾选。
