# 项目交接

- 更新时间：2026-10-11（Asia/Shanghai）；分支 `dager-rank-and-experiment-records`，基于 `main` 的 `73ebac2`，尚未合并回 `main`。
- 2026-10-05 至 10-10 的工作已按用户要求拆分提交：日志 loss 修复 `e9a05eb`，筛选研究 `7be400e`（[[dager-filter-phase]]），图像排序诊断 `de95a9f`（[[token-image-ordering]]），rank 对比 `4bae2c8`（[[dager-lora-rank]]），以及实验记录与本地文档站点。
- 筛选研究的打分函数没有注册为攻击，`dager_adapted` 的默认行为不变。
- 2026-10-10 rank 对比已全部完成。首次 OOM 后修复日志计算，用户明确允许重启；两个模型和全部配对重构、评估均完成（[[dager-lora-rank]]）。
- 日志修复后的全量 CPU 回归为 293 通过、3 跳过，Ruff 与本地文档构建通过；新 GPU 运行已成功保存首轮训练后快照。

接手顺序：先读 `AGENTS.md`，再读本文件，然后看实验索引 `docs/experiments/README.md`。每个实验一页，包含条件、数字、产物和局限，也可以用 `mkdocs serve` 浏览。开始前先确认状态：

```bash
git status --short && git log -8 --oneline
ps -eo pid,etime,cmd | rg 'examples.run_attack|utils.run_cmds'; nvidia-smi
```

## 当前状态

- **目标**：评估联邦微调 VLM 时的隐私泄露，覆盖不同架构、数据集、联邦算法、FedVLMBench 微调策略、攻击者知识，以及视觉、文本、语义和代价指标。先打通一条完整路径：SLAKE + LLaVA-1.5-7B。
- **阶段**：方法和协议的验证与诊断。端到端流程已经跑通，DLG、IG、TAG 和 `dager_adapted` 都在 SLAKE / LLaVA 上运行过。
- **图像侧（IG）**：
    - 所有测试设置下，重建都停留在噪声水平（[[ig-parameter-sweep]]）。
    - 梯度回放是精确的；问题在于匹配目标更偏好非真实图像，从真实图像附近出发也会漂移开（[[ig-failure-diagnosis]]）。
    - 训练后的 F-L 只出现弱的回复力，平衡点约为 24 dB（[[ig-strategies-and-rounds]]）。
- **可区分性**：
    - 匹配损失能在有限候选集上给出有用的全局排序，但局部像素梯度几乎与指向真实图像的方向正交（[[gradient-discriminability]]）。
    - 文本完全未知时，训练后的模型失去了这种排序（[[text-unknown-discriminability]]）。
    - 图像排序尚不能作为通用 token 成员判据；LoRA-A 的答案排序出现值得独立验证的次要信号，但仍使用真实图像构造候选集（[[token-image-ordering]]）。
- **文本侧**：
    - 已知图像和长度时，TAG 没有恢复出任何问题内容（[[tag-text-only-public-image]]）。
    - `dager_adapted`（只重建文本，已知 token 长度；原版 `dager` 仍未实现）的第一阶段 token 信号可以复现：residual 打分让 Recall@100 从 0.61 升到 0.98（GT）和 0.96（Random）。
    - rank 8 即使候选全覆盖仍无法恢复完整问答（[[dager-multisample-n10]]）；后续提高 rank 后，已知图像时完整问答恢复改善，随机图像下仍困难（[[dager-lora-rank]]）。
- **2026-10-05 筛选改进**：
    - 三种改进都不能稳定替代现有筛选器：稀疏字典排序只在已知图像时有效；公开图像统计加权和逐 token 校准都没有通过独立验证（[[dager-filter-phase]]）。
    - 保留现有筛选器。
    - 包含这批代码及图像排序诊断时，最近一次 CPU 回归为 289 通过、3 跳过；Ruff 无报错。

## 优先级

按优先级从高到低排列：

- **DAGER 文本方法**（用户 2026-10-05 的决定）：
    - 用户指定的 rank 对比已完成；下一研究方向是独立验证 rank 32，并诊断其候选全覆盖后失败的答案和随机图像下的问题序列（[[dager-lora-rank]]）。
    - 先做第一阶段的 token 筛选，再做句子搜索；目标是在可控的候选数量下完整覆盖私有 token 集合，同时统计漏检和误报（见 [[dager-multisample-n10]] 中的事后审计）。
    - 用户 2026-10-06 指定的首轮图像排序诊断已完成（[[token-image-ordering]]）。若继续该方向，先冻结 LoRA-A 的答案打分规则，在新的多 token 答案上独立验证，并对照平均损失；暂不修改默认筛选器。
    - 序列方向仍待诊断改编版的 beam search：结果提交后检查真实前缀能否保留，以及每一层的排名。
    - 再实现一个单独命名、忠于原论文的阈值解码器作对照。
    - LoRA-A 梯度为零时的初始化仍明确不支持。
- **利用可区分性的排序**：查看按模块和候选类型拆分的结果，测试粗粒度或无需梯度的搜索能否在不依赖私有参考的情况下利用这个排序。
- **IG 随机起点基线**：用 float32 候选图像重跑（`attack.image_dtype`，2026-09-24 起为默认值）。参数扫描用的是 bf16 候选图像，引用前需要重跑。
- **TAG 文本侧后续**：跑一组配对实验，`image_question_known` 加已知答案长度，覆盖 F-C、F-L、F-CL 的第 0、10 轮，用来区分联合搜索失败和文本目标没有信息量这两种解释。`run_yaml/slake_llava_image_question_known.yaml` 只跑过一个 20 次迭代的 pilot（`outputs/diagnostics/iqk_pilot`）。
- **策略与轮次结果在写进论文之前**：在训练好的 F-C 快照上跑 IG 网格（2026-09-29 已训练完，攻击网格还没跑）；用 10–20 张图像代替 3 张；增加种子和训练轮次。
- **只有当更新中出现可恢复的图像或文本信号后，才扩展实验**：F-2stage；`private`、`question_known` 和已知长度等条件；正式运行时启用 LPIPS、CLIP 和私有文本指标。

## 假设（未验证）

- IG 从近真值起点漂移开，是因为目标函数偏好其他图像，而不只是 sign step 的噪声下限（每步每像素 ±lr）。尚未做的对照：同样的起点和 sign step，但匹配一个错误的更新；以及更小的学习率。
- F-L 的回复力会随训练轮次继续增强（目前只看了第 0、5、10 轮）。
- F-CL 第 10 轮的回落反映的是训练后 connector 改变了模型状态，而不是偶然（n=3；现在已有训练好的 F-C 快照，可以帮助区分）。
- 限制可区分性的是更新中与图像无关的成分：纯噪声图像的梯度与真实梯度的余弦相似度仍有 0.4–0.7，训练后更高。
- DAGER 漏掉答案末尾的子词，是因为答案最后一个输入位置的 EOS 监督偏弱（见 [[dager-multisample-n10]] 中的审计部分）。

## 运行中与待处理的任务

- 2026-10-10 用户授权的新运行 `outputs/diagnostics/dager_rank_gpu5_20261010_v2/` 已完成；worker PID 745473、watcher PID 745474 均退出。两个第十轮模型位于该目录的 `models/rank-16/round-0010/` 和 `models/rank-32/round-0010/`；完整比较见 comparison.json、analysis.json（[[dager-lora-rank]]）。
- 2026-10-10 rank 对比的 worker PID 743410、watcher PID 743411 已退出。产物 `outputs/diagnostics/dager_rank_gpu5_20261010/` 保存失败记录与 rank 16 第 0 轮模型；没有训练后快照或重构结果。不要把 progress.json 的 train 阶段视作仍在运行（[[dager-lora-rank]]）。
- 2026-10-06 图像排序诊断已在 GPU 5 完成，worker PID 448675 与 watcher PID 448676 均已退出。产物根目录 `outputs/diagnostics/token_image_ordering_gpu5_20261006/` 中存在 `complete.json`；回放和文件哈希复核通过（[[token-image-ordering]]）。
- 2026-10-05 在 GPU 5 上的所有 worker 和 watcher 都已退出。
- `outputs/train_stealth_gpu5_70g.pid` 记录的 PID 1687625 已不存在。
- 2026-10-10 检查时另有 `train_stealth.py --model_size xlarge --batch_size 30 --gpu_util 0.4`（PID 692001）在运行，不属于本项目。GPU 5 启动前已有约 30 GB 显存占用，本轮未修改该进程。
- `run_yaml/slake_llava_ig_iteration_occ.yaml`（100,000 次迭代的诊断）在第 15150 次迭代时中断，调度器里显示 `running` 的那一行已经过期。
- 显存参考：
    - IG 攻击峰值约 29–32 GB（F-C、F-L、F-CL，bf16 受害模型，二阶梯度）。
    - batch 2 的 `train` 约 20 GB。
    - LLaVA 的 DAGER 打分约 16 GB。
    - 训练快照网格使用 `min_free_mib: 36000`，更早的调度使用 50000。
- 长时间任务用 `setsid nohup … & disown` 脱离启动，并用 `outputs/diagnostics/analysis_scripts/watch_*.sh` 监视；watcher 在任务完成、失败、出现 traceback 或 OOM、调度器退出时结束。

## git 记录不了的信息

- `Observation` 当前是 schema v5（支持声明公开图像）。
    - `outputs/slake_llava_ig_parameter_sweep/`、`outputs/diagnostics/step0` 和 `step1_pilot` 中的捕获是 v4，会被 `attack` 和 `diagnose-gradient` 拒绝；要复用需要重新捕获（确定性，几分钟）。
    - `outputs/diagnostics/reference_init*` 和训练快照网格是 v5。
    - 所有已报告的数字仍然有效。
- 生成 `outputs/diagnostics/loss_curves/*` 的分析代码不在仓库里，副本放在 `outputs/diagnostics/analysis_scripts/`（损失曲线、α 与受害模型精度、CLIP 特征敏感性、参数统计、绘图、逐任务对比、watcher）。用于论文图之前，需要先并入 `evaluation/`。
- `docs/ppt/02-preliminary-results.png` 用的是示意性的参考图像，最终版本中要换成 `docs/ppt/slake-reference-*.png`。
- 标为 `private-tokens` 的实验页引用了私有 token 片段；文档站点只能在本地查看，不能原样发布。

---
在重要提交、实验阶段结束或方向改变后更新本文件。实验结果写在 `docs/experiments/<date>-<id>.md`（写完运行 `python -m utils.experiment_index`）；本文件只保留当前状态、优先级、假设、任务和注意事项。用 `[[id]]` 链接实验，把测量结果和假设分开写，并删除已经过时的内容。
