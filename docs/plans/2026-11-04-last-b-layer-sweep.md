# LoRA-B 全层读出扫描

用户确认扫描所有层。先固定本计划，再对原 SLAKE30 的 32 层 down_proj.B、o_proj.B 共 64 个位置评分，确定当前读出方法的候选位置。只做候选筛选，不运行序列搜索、不合并多个位置、不提交 Git。本机与 WN-A100 的本项目任务合计最多两张 GPU，本阶段只用一张完全空闲卡；无空闲时仅 CPU 排队，不共享忙卡。

## 研究问题

- 当前输出词方向近似在哪些层和模块有效，答案信号是否形成连续高值区域，而非只有一个孤立最优点？
- 同一 80/20 补充规则下，哪些位置优于 L32 down、L32 o、L31 down，提升是否同时出现在 Noise 与 Top-k？
- 这些位置能否提供问题侧候选，还是仍只改善答案词？

## 固定设置

| Item | Setting |
|---|---|
| Model | 原 SLAKE LLaVA-1.5-7B，F-L，Round 10，rank 32、alpha 64；原 bf16 模型与上传 |
| Uploads | 原 SLAKE30、短中长各 10 QA；None、Noise ratio 0.01、Top-k 10%，每种 30 份，不重捕获或重新加噪 |
| Positions | L1–L32 的 down_proj.B 与 o_proj.B；代码层号 0–31，均为 4096×32。完整参数名在准备时与 Observation 核验 |
| Readout | u_w=γ⊙W_head[w]；s(w)=‖G_Bᵀu_w‖²/(‖u_w‖²‖G_B‖²_F)，高分优先；分析 float64，不改符号或做层特异校准 |
| Controls | 原 Residual、原公开 LM marginal、每个位置同防御下固定循环错配一份上传；三个已测试位置逐分数与候选复现 |
| Budget | 每个位置单独 informative Top-100；固定补充为原 Top-80 再补该位置 20 个不重复 token。共同 9 个 ambiguous 另保留，总数始终 109 |
| Metrics | 主指标为 80/20 补充后的 Answer recall，同时报告 Answer FullCoverage；辅助报告单独 AP、Question 和 Joint 覆盖。融合 AP 留空，Joint FullCoverage 不是序列 EM |
| Image | 新读出不接收图像，GT/Random 相同；各自的原 Residual 与固定补充保留图像条件 |
| Commitment | 90 份上传×64 位置，共 5760 组词表分数；全部匹配、错配、基线及固定补充候选一并保存后才评价真实 token |
| Output | outputs/diagnostics/last_b_layer_sweep_20261104/，冻结 source-sweep-v1 和输入/历史分数哈希 |

公开加权词向量及其范数只缓存一次，使用与原公式相同的乘法、平方、求和和分母；不改变归一化顺序。三个历史位置要求词表分数 max diff≤1e-12、候选集合及指标复现。CPU 评价缓存公开排序，以避免每个字段重复排序；数值定义须与既有评价一致。

## 预设分析与选层规则

- 全部层和模块分别报告 None、Noise、Top-k，不按最优层删去其余结果；热力图显示层号与模块，单独读出和固定补充分图。
- 同一位置的匹配上传与错配上传比较，避免把常见词偏好误认为目标特异信号。
- 分别报告 Noise、Top-k 的候选最优位置，保留 GT/Random 全部指标；不按每条样本的真值选择层或补充候选。
- 如需选出后续确认候选，固定以 Random 的 Noise、Top-k 补充 Answer recall 均值降序，再按两者较小值、答案完整覆盖均值、较深层和模块 down 在前打破平局；分别保留每个模块的第一名。None 只作为损伤对照，不参与主排序。
- 主排序只用于同批开发样本的发现，不能称作实际攻击的逐样本选层规则。两个模块候选必须在新样本确认后，才能宣称泛化最优；本轮不自动采集新私有上传，独立确认是明确的后续阶段。
- 显示与既有 L32 down、L32 o、L31 down 的差别，以及最优点附近的曲线；某一浅层差不代表该层没有隐私信息，只说明当前近似在该处表现差。

## 执行与停止条件

- [x] 验证 64 参数映射、缓存打分与原公式、向量化 AP 与原评价、固定预算及先评分后真值边界；冻结代码。六项新测试及既有位置/读出共 20 项通过，Ruff 通过。
- [x] 单卡 detached 启动，记录主机、GPU UUID、环境、PID、命令；核验首份完整 64 位置评分及三个历史位置复现。
- [x] 全部 5760 组词表分数和固定候选保存后统一评价，不重训、不跑搜索或多位置融合。
- [x] 按预设规则导出全部层的热力图、曲线、主排序和候选层；明确属于同批探索，不是独立确认。
- [x] 独立审计哈希、同预算、历史复现及指标；结果扩展 [[dager-upload-defense]]，更新交接与索引，核验资源释放。最终 44 项定向与文档索引测试、Ruff、21 份实验记录与本地产物、MkDocs 严格构建和 23 个发布/代码哈希通过。

64 个位置全部测试完成或发生明确的阻断性异常即结束；所有异常保存失败状态并退出，不自动 OOM 重试或更改 dtype、预算、知识条件、层范围。保留所有既有产物和其他会话改动。

## 运行记录

- source-sweep-v1 SHA256：3222549a7254438dc0198804b70f2f6d3867ac714d5ffc265994ba2e53020f83，base commit 2fd25b6，gia 解释器。准备代码与环境在本机 NAS；不改生产攻击或共享环境。
- 启动前 WN-A100 GPU 0 恢复空闲，0 MiB、利用率零，UUID GPU-25dab3f9-1eda-bcfd-291e-9ca1ca3b673b；仅使用这一张卡。controller 2386、GPU worker 2391、watcher 2396，命令在 controller.json，日志 setup/score.log、setup/evaluate.log。
- Launcher：ssh zx-a2 'cd /home/zx/nas/GitRepos/giavlm && bash outputs/diagnostics/last_b_layer_sweep_20261104/setup/launch.sh'；十分钟状态日志与三十秒显存巡检由 watcher 执行，完成、失败或控制器死亡时退出。
- 首份 64 位置词表分数、形状、有限范围和哈希通过；三个历史位置分数最大差均为零，没有在看真值后改方法。
- 全部正常完成，CPU 评价 2487；GPU worker、controller、watcher 和 CPU 评价均已退出，两个阶段 returncode=0。墙钟 179.26 秒，峰值 allocated 16.65 GiB，最终核验 GPU 显存零、利用率零；没有待运行阶段。
- 独立审计通过：270 份历史分数差零、2160 组历史候选和原指标复现、全量预算 109、271 个评分/候选哈希、768 次原实现候选重算及所有聚合/配对/排序核验。来源与导出文件见 last-b-sweep-manifest.json。

## 结论与候选冻结

- 按预设跨防御规则，down 模块第一名为 L31 down（均值 0.8482），o 模块第一名为 L32 o（0.8462）；它们仍是前轮的两个位置。L32 down 排第 13，均值约 0.8000。
- Top-k 的单项补充召回由 L29 o 略高（0.8738），但其 Noise 为 0.8091、答案完整覆盖也不全面占优；不按单一最有利指标替换预设选择规则。
- 高值区域集中于后部相邻层，并非严格单调。全 64 位置也没有修复问题侧漏检，Random 的 Joint FullCoverage 最大仍为 Noise 0/30、Top-k 1/30。
- 后续确认候选固定为 L31 down 与 L32 o。当前只有同批探索，不自动采集新私有上传、做独立确认、多位置融合或序列搜索，不提交 Git。
