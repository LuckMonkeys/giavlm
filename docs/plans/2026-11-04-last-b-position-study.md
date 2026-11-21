# LoRA-B 读出位置对照

用户授权验证三个预先选定的位置，沿用[末层 B 读出](2026-11-04-last-b-token-readout.md)的同一 SLAKE30 上传、打分、词方向和候选预算。先写计划、冻结源码，再使用一张空闲 GPU；本机与 WN-A100 的本项目任务合计最多两张。只验证 token 候选，不运行序列搜索、不提交 Git。

## 研究问题与假设

- 末层 down_proj 距答案损失最近，已有正向答案证据；本轮首先逐分数和候选集合复现它，作为实现控制。
- 末层 o_proj 的输出仍位于 residual hidden 坐标，但误差反传多经过一个逐位置 MLP；可能保留或补充答案方向。
- 倒数第二层 down_proj 还要经过一个完整 Transformer block，后续注意力会混合位置。当前输出词方向近似可能变差，但不能预先假设性能随深度单调变化。
- 各位置都使用相同的最终 RMSNorm 修正词向量。相同维数并不证明方向完全一致；本轮正是检验这个近似的适用边界，不用私有隐藏状态修正。

## 固定设置

| Item | Setting |
|---|---|
| Model | SLAKE LLaVA-1.5-7B、F-L、Round 10、rank 32、alpha 64，原 bf16 快照 |
| Uploads | 原 SLAKE30，None、Noise ratio 0.01、Top-k 10%，每种 30 份；不训练、重捕获或重新加噪 |
| Positions | L32 down_proj、L32 o_proj、L31 down_proj；L32 对应代码 layers.31，L31 对应 layers.30 |
| Parameters | 各模块 lora_B.default.weight，均为 4096×32；不加入 Q/K/V、gate/up 或其他层 |
| Readout | u_w=γ⊙W_head[w]；s(w)=‖G_Bᵀu_w‖²/(‖u_w‖²‖G_B‖²_F)，高分优先；分析 float64 |
| Controls | 原 Residual；上轮冻结的公开 LM marginal；每个位置同防御下循环错配一份上传 |
| Budget | informative 100，另保留原共同 9 个 ambiguous token，实际 109 |
| Fixed supplement | 每个位置分别按原 Top-80、该位置 20 个不重复 token 构造候选；LM 对照配额相同。不把三个位置相互合并，不选最佳层后只报该层 |
| Metrics | Answer、Question、Joint 的候选 recall、Informative AP、FullCoverage；融合不定义全词表排序，AP 留空；Joint 不是序列 EM |
| Boundary | 评分函数只读取 Observation 张量与公开模型；全部三个位置的分数、阴性对照及固定补充候选保存后才评价真实 token |
| Reproduction | L32 down_proj 必须复现上轮全部 90 份词表分数（max diff≤1e-12）和候选集合；原 Residual、LM prior 及 80/20 补充也复现既有结果 |
| Output | outputs/diagnostics/last_b_positions_20261104/，固定源码、输入和参考分数哈希 |
| Resource | 预检 WN-A100 GPU 1 空闲，UUID GPU-6a309ac1-08fa-062a-b449-e8de39ba4d09；只使用一张，启动前复查。无空闲时仅 CPU 等候，不共享忙卡 |

模块完整名称在 prepare 阶段由 Observation 的公开 parameter_names 精确核验。读出本身不接收图像，GT/Random 的差别只来自原 Residual 及其补充组合。先前 SLAKE30 已用于开发，本轮仍是同批机制诊断，不称独立验证。

## 执行与判定

- [x] 先验证三模块映射、同一能量评分、错配对照、去重预算及评分先于真值边界；冻结代码和所有复用输入哈希。新增五项与原读出共 14 项通过，Ruff 通过。
- [x] 单卡 detached 启动，记录主机、GPU UUID、环境、命令与 PID；核验首份三位置分数及 L32 down_proj 复现。
- [x] 90 份上传×3 位置共 270 组词表分数全部保存；全部位置单独候选及固定补充候选再一起保存，统一评价。
- [x] 分别报告三个位置、各自错配对照与每种防御，不按参考指标更换符号、层、比例或截断。
- [x] 用同样本配对差值检查新位置相对 L32 down_proj 的答案 recall/AP、完整覆盖得失和候选补充差异。只有某一位置有稳定增益，才将它列为后续独立验证候选；本轮不自动进行多位置融合。
- [x] 结果扩展 [[dager-upload-defense]] 的末层 B 读出，导出聚合表图和 SHA256；更新交接、索引并核验资源释放。最终 38 项定向及文档测试、Ruff、21 份记录及本地产物、MkDocs 严格构建和 16 个发布/代码哈希通过。

三位置都测试完成即结束；若新位置无收益，结论是当前末层 down_proj 仍更合适，不扫描更多层。任何异常保存失败状态并退出，不自动 OOM 重试、改精度、预算或知识条件。保留上轮与其他会话产物。

## 运行记录

- source-positions-v1 SHA256：775c0093ddcc2878ff72b92e85dcb078eecfbd1fcfe2fcf550343f27674d0f09；base commit 2fd25b6。源于上轮固定源码，加入新位置入口与测试，运行中不再修改。
- WN-A100 GPU 1、UUID GPU-6a309ac1-08fa-062a-b449-e8de39ba4d09，启动前 0 MiB、利用率零。controller 2254、GPU worker 2259、watcher 2264；gia 解释器，单卡执行，没有使用第二张卡。
- Launcher：ssh zx-a2 'cd /home/zx/nas/GitRepos/giavlm && bash outputs/diagnostics/last_b_positions_20261104/setup/launch.sh'。命令、PID、UUID、输出和各阶段终态在 controller.json；日志 setup/score.log、setup/evaluate.log；watcher 十分钟记录完整状态、三十秒巡检显存，结束或失败退出。
- 全部正常完成，CPU 评价 PID 2353，两阶段 returncode=0；总墙钟 134.68 秒，峰值 allocated 14.42 GiB。controller、worker、watcher 均已退出，最终核验 GPU 显存零、利用率零。
- 独立审计通过：L32 down 的 90 份词表分数差为零、1080 组历史候选和原指标复现、全部预算 109、271 个评分/候选哈希及聚合和配对指标重算。导出十一份聚合表图与审计，来源见 last-b-positions-manifest.json。

## 结论

- L32 o 与 L31 down 在本批答案候选均值上优于 L32 down。Random 的同配额补充在 Noise 下为 0.8265、0.8273，Top-k 为 0.8659、0.8690，原 L32 down 为 0.7757、0.8244。
- L31 down 的固定补充通常略高，但 L32 o 的单独 AP 更高，Noise 的 GT 补充也是 L32 o 略高；没有单一全面最优位置，不把结构距离当成性能保证。
- Joint FullCoverage 仍为 Noise 0/30、Top-k 1/30，只改善答案侧。结果已并入 [[dager-upload-defense]]，不继续扫层、多位置融合或序列搜索，不提交 Git。
