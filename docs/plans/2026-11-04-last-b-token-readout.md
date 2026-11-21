# 末层 B 梯度的答案 token 读出

先记录假设，再复用现有上传验证，不提交 Git、不改生产攻击。使用一张空闲 GPU；本机与 WN-A100 的本项目任务合计不超过两张。最初预检的 GPU 7 在启动前被占用；资源队列随后选到 WN-A100 的空闲 GPU 1（UUID GPU-6a309ac1-08fa-062a-b449-e8de39ba4d09），实际仅用这一张卡，运行记录保留全部变化。

## 假设与边界

当前第零层 A 梯度的公开输入消除后信号很弱；A/B 一致性校正已经通过数值验证，却没有改善防御下的完整候选覆盖。本轮改变信息来源：直接使用最后一层 MLP down_proj 的 B 梯度，寻找答案监督留下的输出词方向，不再校正第零层 A。

精确算术下，G_B=c∑ δ_t(Ax_t)ᵀ，求和只覆盖答案监督位置（含 EOS 的监督位置）。最后一个 down_proj 后没有跨位置注意力，δ_t 经最终 RMSNorm 和公开 LM Head 回传，包含预测分布与真实目标的差。图像仍通过上下文影响隐藏状态；不能假设这份梯度完全独立于图像，也不能保证真实词向量落入梯度子空间。

最终 RMSNorm 的缩放向量记为 γ，候选输出词向量记为 W_head[w]，固定近似方向 u_w=γ⊙W_head[w]。本轮只验证未建模的 RMSNorm 径向项和预测分布项是否仍允许词表排序，不使用真实隐藏状态修正它们。

    s_B(w) = ||G_Bᵀ u_w||² / ||u_w||²

实际评分统一除以 ||G_B||²_F 保持数值尺度；不影响同上传词表排序。高分优先，不做子空间秩选择、真值选符号、相减公开梯度或防御逆变换。Noise、Top-k 同规则；固定层与模块，不在结果上选层。

## 固定实验

| Item | Setting |
|---|---|
| Model | 原 SLAKE LLaVA-1.5-7B，F-L，rank 32、alpha 64，第十轮 bf16 快照 |
| Source | outputs/diagnostics/federated_text_slake30_20261023/；原 30 QA，短中长各 10 条，不重训练、捕获或加噪 |
| Uploads | None、Noise ratio 0.01、Top-k 10%，每种 30 份；现有 GT/Random 上传须核验哈希相同 |
| Module | backend.language_model.model.layers.31.mlp.down_proj.lora_B.default.weight |
| Computation | 原模型/上传 bf16，B 读出与候选排序 float64；分析阶段不反传 |
| Baseline | 复用原完整 Residual 的全部词表分数、ambiguous 与 selected IDs；GT、Random 各自保留 |
| Budget | informative K=100，加原共同 9 个 ambiguous ID；所有方法实际候选数固定 109 |
| Metrics | 分别评价 Answer、Question、Joint 的 token recall、Informative AP 和 FullCoverage；Joint 不是序列 EM |
| Boundary | 评分只读取公开模型和 Observation；全部分数与候选保存后才读取真实 token。新读出本身不接收图像，在 GT/Random 下相同；与各自基线融合后再比较 |
| Output | outputs/diagnostics/last_b_token_readout_20261104_v2/，冻结 source-last-b-v1；首次资源预检失败目录另保留 |

### 对照

- Baseline：原第零层 Residual 排序与候选集。
- Last-B：上式对全部词表评分；特殊 token 排除规则与原方法一致。纯新方法仍保留共同 ambiguous ID，以固定实际候选数。
- LM marginal：公开权重的 text-only 语言先验，固定五个问题模板（下列），每个最多贪婪展开 16 个 token，遇 EOS 结束；汇总各位置的 next-token 概率，先在每个模板内平均，再在模板间平均。不使用私有问题、答案或图像。
- Wrong-upload：仅作延迟评价的阴性对照，把冻结顺序中的下一份上传分数分配给当前样本（循环移位一位），不产生额外攻击分数；检查读出是否比不匹配上传更有目标特异性。不同样本可能有重复答案，因此阴性对照不能保证零命中。

固定 LM 模板：What is shown in the image?；What type of image is this?；Which body part is shown?；Is an abnormality visible?；Describe the image.。以公开 USER/newline/ASSISTANT 片段分别分词构造纯文本上下文；不自动改 prompt、过滤退化结果或补采样。

### 阶段与停止条件

- [x] 数学验证：末层 B 的因果监督位置、RMSNorm 近似方向、显式能量评分与矩阵公式、缩放不变性、同预算和延迟评价边界。新测试 9 项，连同原筛选相关测试共 48 项通过，Ruff 通过。
- [x] 冻结源码、模型元数据与上传哈希；远端单卡 detached 启动，首份产物核验、watcher 退出机制与资源记录。
- [x] 90 份末层 B 分数、1 份公开 LM prior 和全部预设候选保存后，统一评价三个条件和两种图像基线；报告全部三十样本及长度桶。
- [x] 固定预算融合的预设门槛：None 条件 Last-B 的 Answer recall 与 AP 均高于 LM marginal，且至少一种防御下同时高于原 GT、Random 基线与 wrong-upload；上述比较都用三十样本均值。若未通过，不跑融合、序列搜索或继续调方向。
- [x] 若通过门槛，仅验证一个冻结的 80/20 规则：保留原 Top-80，另取 20 个 Last-B 的未重叠 token，共同 ambiguous 另保留。比较与 Baseline、同配额 LM 补充的差异；所有候选再次先提交后评价。同批数据只作机制诊断，不称独立验证。
- [x] 结果并入 [[dager-upload-defense]]，导出无私有 token 的聚合表图、审计及来源；更新索引、交接并核验资源释放。

门槛不通过即完成本轮，记录该近似读出未验证有效，不转而扫描更多层、阈值或评分符号。所有异常保存状态后退出，不自动 OOM 重试或改 batch/dtype/知识条件。核心目标是新增答案证据，不能将答案侧指标改善解释为完整问答已恢复。

## 执行记录

- base commit 2fd25b6；冻结源码 SHA256 为 885c69d9edb5160ba2ca5d22dd0757144a611d7f1cd5d26c2921afac11059df5。环境保持 /home/zx/nas/miniconda3/envs/gia/bin/python，所有代码和环境准备在本机 NAS。
- 首次 WN-A100 controller 1817 在 GPU 7 资源预检失败后退出，没有评分 worker 或 CUDA 实验；watcher 1823 退出。失败目录 outputs/diagnostics/last_b_token_readout_20261104/ 保留。
- 新目录 outputs/diagnostics/last_b_token_readout_20261104_v2/ 保持同一算法、源码与输入计划，controller 1851、watcher 1857 仅用 CPU 等候一张完全空闲的 WN-A100 卡。选择条件为显存少于 100 MiB、利用率为零；选定后记录编号和 UUID，显式绑定一张，不占用忙卡。
- 启动命令为 ssh zx-a2 'cd /home/zx/nas/GitRepos/giavlm && bash outputs/diagnostics/last_b_token_readout_20261104_v2/setup/launch.sh'；状态 controller.json、日志 setup/score.log，监控 watch-latest.json 与 memory-history.jsonl。尚未选定卡时没有模型进程或 CUDA 上下文。
- 资源队列等候 313.77 秒后，选定 WN-A100 GPU 1，worker 1935 从启动到全部评分候选提交为 122.92 秒，峰值 allocated 14.42 GiB；未使用第二张卡。
- 首份检查器误设词表长度为 32000，实际模型为 32064，controller 报错；评分 worker 已正常写完 90 份分数和全部候选，state-score=completed。原失败状态保留，原父进程未收集退出码，不编造 returncode=0。本机 finish.py 只独立核验并续接 CPU 评价与报告，终态 completion-controller.json，没有重复 GPU 评分或改变算法。
- 本轮远端 controller、watcher 和 GPU worker 均已退出，该 GPU 随后有其他任务，不停止他人进程或宣称整卡空闲；没有启动本机后台续办程序。原候选 180 组、全部 451 个评分及候选哈希、聚合指标和 18 个发布/代码哈希均通过独立审计。

## 完成状态

通过预设答案信号门槛，已进一步验证固定预算融合；这仍是同批机制诊断。

- 结果与表图已并入 [[dager-upload-defense]]；融合门槛、是否执行融合、来源与审计均明确记录。全部未提交 Git。
- Random 条件固定 80/20 Last-B 融合后，Noise、Top-k 的答案 recall 分别为 0.7757、0.8244，答案完整覆盖为 10/30、11/30；同配额 LM 的 recall 为 0.5217、0.6149。Joint FullCoverage 仍为 0/30、1/30，不能解释为完整序列恢复。
