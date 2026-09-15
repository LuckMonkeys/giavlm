# GIAVLM 项目学习路线

记录日期：2026-09-15。

本路线基于整理时已查看的项目实现，用于理解现有代码并检查它是否符合研究预期，不代表所有规划功能已经完成。后续代码变化时，应以源码和重新运行的测试为准。

## 1. 学习目标与方法

围绕四个问题学习：

1. 当前具体实现了什么，哪些功能只是接口或占位？
2. 如何调用入口、组合配置并运行测试？
3. 各阶段的输入输出格式是什么，哪些信息对攻击者公开？
4. 从正确性验证到正式实验，完整流程应该怎样组织？

采用“跑通一个样本 → 追踪数据流 → 理解数学实现 → 验证研究假设”的顺序，不要从目录开始逐个读文件。每个阶段都留下可检查的结果，完成验收后再进入下一阶段。

**第一个学习目标：独立解释一个 tiny 样本从输入到评估结果的全过程，而不是先掌握所有攻击。**

## 2. 建立当前功能地图

先读 [项目结构](../PROJECT_STRUCTURE.md)、[攻击注册表](../attacks/registry.py) 和 [项目约束](../AGENTS.md)。

| 状态 | 内容 |
|---|---|
| 已接通 | Hydra 实验循环、FedSGD/客户端多步 delta、full/LLM-full/LoRA、上传参数掩码、恢复与防御接口 |
| 已有实现 | VLM 适配版优化攻击、图像和文本联合重构、COCO/VQAv2、医疗 VQA 导入、canary 注入及评估 |
| 有条件运行 | LLaVA、BLIP-2、Qwen2.5-VL，以及需要外部权重的先验、LPIPS、CLIP |
| 尚未实现 | 闭式 APRIL、iDLG、DAGER、H3 两阶段恢复、恶意服务器攻击、聚合更新反演等 |

文件存在、方法可运行、论文已复现，是三个不同层次。`*_adapted` 不应被视为原论文的完整复现；`not_implemented` 与 `not_applicable` 也应分开理解。

上次学习路线分析时，在 `/tmp/giavlm-venv` 中运行测试得到 **77 项通过、2 项因缺少 `datasets` 失败**，当时 `pyproject.toml` 也未声明该新增依赖。这是历史观察，不是本文件创建时重新执行的结果。开始学习时应重新检查环境、依赖声明和测试基线，不要修改无关的 Conda 环境。

## 3. 六阶段学习安排

### 阶段一：跑通入口与配置

阅读：[experiment.py](../core/experiment.py)、[主配置](../configs/config.yaml)、[规范入口](../examples/run_attack.py)。

动手任务：展开 Hydra 配置，运行一个 tiny 样本，沿 `protocol_config → ExperimentRunner → capture → attack → evaluate` 追踪调用。

验收清单：

- [ ] 能解释 data/model/attack/defense/fed/knowledge/evaluation 七个配置组。
- [ ] 能区分 Hydra 配置与严格序列化的协议配置。
- [ ] 能解释 `num_runs` 是停止 run ID 的上界，而不是额外运行次数。
- [ ] 能说明默认 `fed.rounds=0`、`model_snapshot` 和断点恢复分别意味着什么。
- [ ] 画出一张调用链图，并标出每一步的输出目录。

### 阶段二：理解数据与公开信息边界

阅读：[data.py](../core/data.py)、[types.py](../core/types.py)、[knowledge.py](../core/knowledge.py)、[commands.py](../core/commands.py) 中的 `capture`。

动手任务：选择一条 JSONL 记录，追踪图像预处理、文本编码、Batch 构建、Observation 导出和私有参考落盘。

验收清单：

- [ ] 建立字段表，标注来源、形状、是否公开、由谁读取。
- [ ] 解释同一图像的多条 QA 如何划分 split/client，以及 capture 的图像去重行为。
- [ ] 比较 private/question_known/text_known 下攻击者可见字段的变化。
- [ ] 确认真实图像、私有 QA、私有长度与参考 mask 不进入攻击接口。

### 阶段三：理解训练损失与联邦观测

阅读：[vlm_wrapper.py](../core/vlm_wrapper.py)、[fl.py](../core/fl.py)、[协议测试](../tests/test_protocol.py)。

动手任务：写出 loss、单步梯度和多步 delta 的公式，再与测试中的优化器更新逐项核对。单步 SGD 应满足 `delta = -lr * gradient`，多步情况需要按更新顺序重放。

验收清单：

- [ ] 能解释 prompt、response、EOS、padding 在 loss 中的作用。
- [ ] 能列出 full、llm_full、lora_llm 的可训练参数和实际上传参数。
- [ ] 能解释上传参数掩码与冻结参数的区别。
- [ ] 确认当前是 SGD 重放，每个 local step 使用一个候选 batch，而不是任意优化器或任意本地训练过程。
- [ ] 能区分客户端多步 delta、多个客户端的聚合结果和联邦训练轮数。

### 阶段四：理解攻击优化

阅读：[engine.py](../attacks/optim/engine.py)、[objectives.py](../attacks/objectives.py)、[priors.py](../attacks/priors.py)。先读 IG，再对比 DLG，最后阅读其他方法组合。

动手任务：追踪一次优化迭代中的候选变量、更新重放、匹配损失、先验、反向传播和候选选择。

验收清单：

- [ ] 能区分待优化图像、软文本变量与不应被优化的已知文本。
- [ ] 能解释为什么梯度反演需要对更新匹配目标求导，以及哪些路径涉及二阶导数。
- [ ] 确认候选选择只依赖公开更新和先验，不依赖参考指标。
- [ ] 能解释迭代数、更新评估次数、restart、时间预算和恢复状态的区别。
- [ ] 比较正常更新、错误更新、random、prior_only 对照；尽量复用同一观测和初始化种子。

### 阶段五：理解指标与结论边界

阅读：[reconstruction.py](../evaluation/reconstruction.py)、[metrics](../metrics/)、[compare.py](../evaluation/compare.py)、[数据与指标测试](../tests/test_data_metrics.py)。

动手任务：先测试完全相同的参考与预测，再测试打乱顺序、错误文本和不同图像。检查多样本匹配和已知文本排除是否符合预期。

验收清单：

- [ ] 能解释 PSNR/SSIM、ROUGE、LPIPS/CLIP 分别反映什么，不能证明什么。
- [ ] 能解释 batch 内联合匹配与逐图、逐文本分别匹配的区别。
- [ ] 检查 canary 是否经过编码和截断后仍存在，以及 recall 的分母如何确定。
- [ ] 区分 `completed`、攻击成功、资源不可用和未实现。
- [ ] 汇总时保留失败状态，并正确使用 prior_only 差值和重复样本的统计单位。

### 阶段六：核对研究预期并进入真实模型

阅读真实模型 [adapters](../core/adapters/)、[HF 架构测试](../tests/test_hf_adapters.py)、医疗数据与 canary 流程，以及 [验证说明](validation.md)。

动手任务：一次只改变一个条件，先做少样本验证，再扩展实验矩阵。小型随机 HF 架构的测试通过，不等于真实预训练模型已经验证。

验收清单：

- [ ] 列出模型、任务、知识条件、更新方式、防御和指标的预期支持矩阵。
- [ ] 对真实模型检查实际 prompt、预处理、损失和 LoRA 插入范围。
- [ ] 检查精确更新重放、候选图像/文本梯度、显存和耗时。
- [ ] 将每个研究预期转化成测试或对照实验，而不只观察最终分数。

## 4. 第一轮调用与测试

在项目根目录执行。先用 CPU fixture，不下载真实模型；下面的输出目录需要尚不存在。重复实验时换一个目录，不要删除已有研究结果。

```bash
PY=/tmp/giavlm-venv/bin/python
$PY examples/run_attack.py --cfg job --resolve
$PY examples/run_attack.py attack.iterations=2 attack.checkpoint_interval=1 output_dir=outputs/learning_01
$PY -m pytest tests/test_protocol.py -q
$PY -m pytest tests/test_data_metrics.py -q
$PY -m pytest tests/test_structure.py tests/test_workflows.py -q
$PY -m core.commands smoke --output outputs/learning_smoke
```

阅读某个模块时运行对应测试；使用 `pytest 路径::测试函数名 -vv` 聚焦单个性质。阶段性完成后再运行 `python -m pytest -q`。

规范实验入口是 `python examples/run_attack.py`；分阶段入口是 `python -m core.commands`。两者的配置用法不同，不要把 staged CLI 的旧平铺配置当成 Hydra 顶层预设。

## 5. 输入输出速查

| 边界 | 格式与检查点 |
|---|---|
| 数据输入 | `samples.jsonl`，一行一条记录；字段包括 `sample_id/image_id/image/question/target/references/task/split/client`，以及可选 `canary` |
| 模型输入 | 图像 `[N,3,H,W]`，范围 `[0,1]`；文本通常为 `[N,L]` token IDs，攻击中可变为 `[N,L,V]` 软分布 |
| 攻击输入 | `capture/public/observation.json`、`update.safetensors`、模型引用；更新为“参数名 → tensor”，不是原始数据 |
| 私有参考 | `capture/private/truth.json`、`images.safetensors`；只供评估使用 |
| 攻击输出 | `attack/result.json`、重构图像和优化检查点；关注 status、reason、questions、targets、costs、history、provenance |
| 评估输出 | `attack/evaluation.json`；关注实验条件、样本匹配、各指标及失败状态 |
| 实验状态 | `experiment.json` 记录各 run 状态；`config.resolved.json` 和每个 run 的 `protocol.json` 用于核对配置 |

这里 `N = batch_size × local_steps`，`L` 是配置中的文本槽位长度，`V` 是词表大小。图像和更新 tensor 使用 safetensors 保存，不应尝试将其当作 JSON 或 pickle 读取。

Python 接口边界不是文件系统隔离。需要更强保证时，应让攻击进程运行在无法读取私有目录的独立用户或容器中。

## 6. 优先核对的研究假设

- **威胁模型**：private 是否仍隐含你不接受的已知信息？text_known 应作为强知识对照，不能与未知 QA 混为一谈。
- **训练协议**：固定块 prompt、文本截断、response-only loss、关闭 dropout 和 LoRA 范围，是否符合目标训练场景？
- **联邦设置**：fedavg 当前观察的是单客户端 delta，不是安全聚合结果；默认 rounds=0 也没有先进行联邦训练。
- **数据采样**：医疗数据会重新按图像划分，capture 会对图像去重；这是否符合目标任务的采样与评估标准？
- **Canary**：注入位置是否仍属私有？实体是否被截断？tiny 的有限词表不能验证真实实体恢复，应使用真实 tokenizer。
- **防御解释**：当前攻击是防御非感知匹配；Gaussian 扰动没有隐私会计，不能直接宣称某个 epsilon/delta 保证。
- **泄漏归因**：与 random、prior_only、错误更新对照比较，区分先验生成能力和观测更新带来的恢复增益。

## 7. 整体验证流程与学习记录

推荐顺序：

**环境基线 → 单元测试 → tiny 端到端 → 单因素对照 → 真实模型少样本 → 正式实验矩阵。**

每发现一个疑点，按下面的表记录。先确定是环境问题、实现错误，还是研究协议选择不符合预期，再决定是否修改代码。

| 我的预期 | 当前实现位置 | 验证实验或测试 | 实际结果 | 类型与后续处理 |
|---|---|---|---|---|
| 例：单步 SGD 的 delta 应等于负学习率乘梯度 | `core/fl.py:simulate_update` | `tests/test_protocol.py:test_one_step_sgd_identity_and_input_gradients` | 运行后填写 | 判断是否需要修改 |

建议最终保留四份学习成果：调用链图、输入输出字段表、实现与预期差异清单，以及能够复跑的最小实验配置。不要在公开学习笔记中保存私有参考、真实患者信息或私有数据路径。
