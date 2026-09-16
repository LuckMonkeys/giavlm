# GIAVLM 项目结构与实现导读

本文基于 2026-09-15 的代码状态，目标是解释项目如何运行，而不只是列出目录。建议先用本文理解一条完整链路，再结合 [学习路线](learning_roadmap.md) 深入具体模块。

## 1. 项目目标与边界

GIAVLM 是一个面向联邦视觉语言模型（FedVLM）的梯度/参数更新反演基准。它把实验拆成五个有明确边界的阶段：

1. 准备并划分图文或 VQA 数据；
2. 构造模型，并可选地进行若干轮联邦训练；
3. 模拟客户端本地更新，生成攻击者可见的公开观测；
4. 从公开观测重构图像和私有文本；
5. 在攻击结束并落盘后，使用私有参考进行评估。

最重要的设计约束是：攻击模块只接收 `Observation`，不能接收真实图像、私有文本、样本路径或参考指标。真实参考只存在于 `capture/private/`，由 `evaluation/` 在攻击产物提交后读取。

当前项目中的 `*_adapted` 是针对统一 VLM 协议实现的攻击适配版，不表示完整复现原论文。未实现的方法会返回 `not_implemented`；不适用于当前协议的方法返回 `not_applicable`。

## 2. 总体执行链

```text
Hydra YAML
    │
    ▼
protocol_config: DictConfig -> 严格 dataclass 协议
    │
    ▼
ExperimentRunner._prepare
    ├── 准备/读取 samples.jsonl
    ├── 校验样本数量、run ID 和恢复签名
    └── 准备共享初始模型或联邦训练快照
    │
    ▼
每个 run-NNNNN
    ├── capture
    │   ├── 加载一个客户端的私有 Batch
    │   ├── 计算 gradient 或 client_delta
    │   ├── 应用上传参数掩码和 defense
    │   ├── 写 capture/public
    │   └── 写 capture/private
    ├── attack
    │   ├── 只加载 capture/public
    │   ├── 重放候选客户端更新
    │   ├── 优化候选图像/文本
    │   └── 写 attack/result.json 和重构图像
    └── evaluate
        ├── 加载 attack 产物
        ├── 此时才加载 capture/private
        └── 写 attack/evaluation.json
```

入口层负责组织实验，算法层分别由 `core/`、`attacks/`、`defenses/` 和 `metrics/` 持有，避免把数据准备、攻击和参考评估混在一个对象里。

## 3. 项目入口

### 3.1 规范入口：完整 Hydra 实验

规范入口是 [examples/run_attack.py](../examples/run_attack.py)。它本身只处理源码目录导入，然后调用 [core/experiment.py](../core/experiment.py) 中的 Hydra `main`：

```bash
python examples/run_attack.py --cfg job --resolve
python examples/run_attack.py attack.iterations=2 attack.checkpoint_interval=1 \
  output_dir=outputs/architecture_demo
```

安装项目后也可以使用等价的控制台命令：

```bash
fedvlm-gia attack.iterations=2 output_dir=outputs/architecture_demo
```

这个入口自动串联模型准备、capture、attack 和 evaluate，适合正式单条件实验与 Hydra sweep。

### 3.2 分阶段入口：调试某一个步骤

[core/commands.py](../core/commands.py) 提供 argparse staged CLI：

```bash
python -m core.commands prepare-data ...
python -m core.commands train ...
python -m core.commands capture ...
python -m core.commands attack ...
python -m core.commands evaluate ...
python -m core.commands report ...
```

它适合学习和调试，因为每一步的输入输出都需要显式传入。该入口读取 `core.config.Config` 格式的平铺 YAML/JSON，而不是七组 Hydra 配置。不要把两种配置接口混用。

其他 staged 子命令包括：

- `inject-canaries`：重写准备好的 manifest，注入合成 PII；
- `doctor`：输出环境和方法能力，可选做梯度重放/二阶导探针；
- `smoke`：运行 tiny 端到端组合；
- `utility`：评估模型正常 VQA/Caption 能力；
- `suite` / `run-suite`：生成和执行旧式大实验矩阵。

### 3.3 批量 Hydra 入口

[utils/run_cmds.py](../utils/run_cmds.py) 读取 `run_yaml/*.yaml`。默认只打印即将执行的 argv，加入 `--execute` 后顺序运行：

```bash
python -m utils.run_cmds --cmd-config-yaml run_yaml/tiny_smoke.yaml
python -m utils.run_cmds --cmd-config-yaml run_yaml/tiny_smoke.yaml --execute
```

调度器不使用 shell 拼接，也不会启动 GPU 占用进程。指定 `--gpu-ids` 时，它用 `nvidia-smi` 检查显存并设置 `CUDA_VISIBLE_DEVICES`。

## 4. Hydra 配置系统

### 4.1 七个配置组

[configs/config.yaml](../configs/config.yaml) 组合以下配置组：

| 配置组 | 主要含义 | 典型字段 |
|---|---|---|
| `data` | 数据来源与采样位置 | `name/task/manifest/split/client/offset` |
| `model` | VLM 及加载方式 | `family/checkpoint/revision/device/dtype/image_size` |
| `attack` | 攻击与资源预算 | `name/text_method/iterations/restarts/lr/seconds` |
| `defense` | 上传前变换 | `name/max_norm/noise_multiplier/ratio` |
| `fed` | 本地更新和联邦训练 | `mode/observation/batch_size/local_steps/lr/rounds` |
| `knowledge` | 攻击者知识 | `private/question_known/text_known` |
| `evaluation` | 可选评估器 | `lpips/clip/clip_model/clip_revision` |

例如：

```bash
python examples/run_attack.py \
  data=vqa_rad model=qwen2_5_vl_3b fed=fedavg fed.mode=lora_llm \
  attack=ig_adapted knowledge=private defense=none \
  data.manifest=/path/to/vqa_rad/samples.jsonl
```

顶层预设按 `<data>_<model>_<attack>_<knowledge>.yaml` 命名，可以通过 `--config-name` 选择。

### 4.2 两层配置

运行时配置是 Hydra `DictConfig`，包含数据路径、输出目录、防御和恢复控制等编排信息。`protocol_config()` 会把它转换成 [core/config.py](../core/config.py) 中的严格 dataclass：

```text
Hydra model.checkpoint  -> ModelSpec.name
Hydra data.task         -> TrainingSpec.task
Hydra knowledge.name    -> TrainingSpec.knowledge
Hydra attack.name       -> AttackSpec.method
Hydra fed.*             -> TrainingSpec.*
```

严格协议由 `ModelSpec`、`TrainingSpec`、`AttackSpec` 和 `EvalSpec` 构成。所有枚举型配置选项集中定义在 `core/config.py` 顶部的有序 tuple 中，便于统一查看和修改。`core.config.validate()` 只负责所有实验路径共享的核心约束，并按 model/training/attack/cross-component 四层组织；dtype、device placement、LoRA 参数、patch 尺寸以及 checkpoint/prior interval 等实现细节，由模型 Adapter 或攻击引擎在实际消费处校验。落盘的 `run-NNNNN/protocol.json` 使用这一格式。

安全聚合攻击目前没有接入完整实验入口；`fed.secure_aggregation=true` 会明确报错，而不是误跑单客户端攻击。

## 5. 核心模块

### 5.1 数据：`core/data.py`

[core/data.py](../core/data.py) 只在私有侧使用，负责：

- 生成 tiny synthetic fixture；
- 规范化 COCO Captions 与 VQAv2；
- 通过 Hugging Face `datasets` 规范化 VQA-RAD 和 SLAKE；
- 按 canonical `image_id` 确定性划分 `train/tune/eval` 和 client；
- 注入合成姓名、MRN 和出生日期 canary；
- 读取 manifest，并把图像/文本转换成 `Batch`。

manifest 是 JSONL，每行至少包含：

```json
{
  "sample_id": "vqa_rad:train:42",
  "image_id": "medvqa:0123456789abcdef",
  "task": "vqa",
  "image": "/absolute/path/to/image.png",
  "question": "What abnormality is visible?",
  "target": "cardiomegaly",
  "references": ["cardiomegaly"],
  "source": "vqa_rad",
  "split": "train",
  "client": 3
}
```

医疗数据没有可靠的上游图像 ID，因此代码对解码后的像素内容求 SHA-256。同一图像的多条 QA 会共享 `image_id`，不会跨 split 或 client。`load_batch()` 使用 EXIF 校正、RGB 转换和居中正方形裁剪，输出 `[0,1]` 范围的 `[N,3,H,W]` tensor。

注意：`capture` 使用 `unique_images=True`，同一个 run 不会选择同一图像的多条问题。这是一个具体实验假设，需要确认是否符合目标论文协议。

### 5.2 数据结构：`core/types.py`

[core/types.py](../core/types.py) 定义四个核心结构：

| 类型 | 作用 |
|---|---|
| `Batch` | 私有图像、问题 token 和回答 token；支持按 local step 切片 |
| `Observation` | 攻击者可见的模型协议、上传 tensor 和授权公开文本 |
| `Support` | `supported/not_applicable/not_implemented` 能力结果 |
| `Reconstruction` | 重构状态、图像、问题、回答、成本、历史和来源 |

若 `N = batch_size × local_steps`，则主要 tensor 形状是：

```text
Batch.images                [N, 3, H, W]
Batch.questions             [N, question_length]
Batch.targets               [N, target_length]
私有候选文本 logits          [N, length, vocabulary_size]
Observation.tensors         {parameter_name: parameter-shaped tensor}
```

`Observation.validate()` 是隐私边界的关键检查：

- `private` 不允许出现任何公开问题/答案及其 token IDs；
- `question_known` 只允许 VQA 问题公开；
- `text_known` 允许问题和目标文本公开；
- caption 不能携带私有问题字段；
- 所有上传 tensor 必须有限且非空。

### 5.3 模型：`core/vlm_wrapper.py` 与 `core/adapters/`

[core/vlm_wrapper.py](../core/vlm_wrapper.py) 定义抽象 `VLMAdapter`，统一以下行为：

- 固定长度编码、EOS/padding 规范化；
- 图像和文本构造 `Batch`；
- full、仅语言模型 full、LoRA 三种可训练参数配置；
- response-only 自回归交叉熵；
- 贪心生成、模型 fingerprint 与结构描述。

`build_model()` 根据 `ModelSpec.family` 创建：

| family | Adapter | 视觉路径 |
|---|---|---|
| `tiny` | `TinyAdapter` | 小型卷积/patch fixture，用于离线正确性测试 |
| `llava` | `LlavaAdapter` | `get_image_features()` |
| `blip2` | `Blip2Adapter` | vision encoder → Q-Former → language projection |
| `qwen2_5_vl` | `QwenVLAdapter` | 官方风格 patchification → visual encoder + mRoPE |

[core/adapters/hf.py](../core/adapters/hf.py) 负责共享的 Hugging Face 加载、tokenizer、图像标准化和固定 prompt 拼接。当前训练格式是项目自定义的固定块协议：视觉嵌入、问题和回答片段被直接拼到 `inputs_embeds`，不是逐模型调用其原生 chat template。

训练 loss 只覆盖 target token，包含 EOS，排除 padding 和 EOS 之后的位置：

```text
L = Σ_i mask_i · CE(target_i, logits_i) / Σ_i mask_i
```

这使真实 token 和攻击中的软 token 使用同一条可微路径，但也意味着结果只代表该固定训练协议。

LoRA 只注入语言侧 attention projection 中名为 `q_proj/k_proj/v_proj/o_proj/out_proj` 的线性层，dropout 被设为 0，模型处于 eval 模式。这些都属于实验定义，而不是所有 FedVLM 训练的默认行为。

### 5.4 客户端更新与服务器聚合：`core/fl.py`、`core/aggregation.py`

[core/fl.py](../core/fl.py) 使用 `torch.func.functional_call` 重放本地 SGD。

FedSGD 单步观测：

```text
g = ∇θ L(B; θ₀)
upload = selected(g)
```

客户端多步 delta：

```text
θₜ₊₁ = θₜ - lr · ∇θ L(Bₜ; θₜ)
Δθ = θ_E - θ₀
upload = selected(Δθ)
```

一个 `Batch` 被按 `batch_size` 切成 `local_steps` 个连续片段，每步恰好消费一个片段。当前未模拟 momentum、Adam、学习率调度、重复 epoch 或 DataLoader shuffle。

`TrainingSpec.upload_parameters` 是 `fnmatch` 模式列表。算法计算本地 update 后把模式解析成显式参数名，再只上传这些 tensor；未命中的模式会报错。攻击重放使用同一算法，因此只返回并匹配公开子集。

`TrainingSpec.algorithm` 是唯一的高层联邦算法字段。[core/aggregation.py](../core/aggregation.py) 中的算法对象统一决定客户端计算与上传、加权聚合以及全局模型更新，同时仍把这些阶段保留为可检查的方法。`fedsgd` 上传并聚合单步梯度，再应用 `-lr`；`fedavg` 上传并聚合本地模型 delta，再直接加到全局参数。显式 factory 是后续加入其他联邦算法的唯一入口。

`simulate_secure_aggregation()` 只是同一模型 fingerprint 下的数值平均工具，不是密码学实现，也没有接入聚合反演攻击。

模型与观测使用 JSON + safetensors 保存。加载时会检查模型 fingerprint、观测元数据 digest、更新文件 SHA-256 和参数名集合。

## 6. Capture 与威胁模型

`commands.capture()` 的顺序是：

1. 从 manifest 选定 `client/split/offset` 的 `N` 个唯一图像；
2. 加载指定模型快照，或新建初始模型；
3. 计算 gradient/client delta，并应用上传参数掩码；
4. 在 update 上应用 defense；
5. 将公开观测和私有参考写入不同目录。

不同知识条件对攻击变量的影响：

| 条件 | 图像 | VQA 问题 | 回答/Caption |
|---|---|---|---|
| `private` | 待恢复 | 待恢复 | 待恢复 |
| `question_known` | 待恢复 | 精确公开 | 待恢复 |
| `text_known` | 待恢复 | VQA 中精确公开 | 精确公开 |

`AdversaryKnowledge` 目前只接受 honest-but-curious、未知模板的条件。恶意服务器和已知模板协议没有在这一入口中实现。

防御发生在更新计算之后、公开观测保存之前：

- `none`：原样复制；
- `clipping`：对整个命名 update 做全局 L2 clipping；
- `gaussian_dp`：客户端 clipping 后加入 Gaussian noise，但没有隐私会计；
- `topk_sparsify`：每个 tensor 保留绝对值最大的坐标，仍以稠密形状保存；
- `sign_sgd`：只保留符号；
- `token_obfuscation/safe_template`：尚未实现，直接报错，避免错误标记实验条件。

当前攻击不会显式建模防御变换，而是直接令候选原始更新匹配防御后的观测。因此结果中会记录 `attack_policy = defense_unaware_raw_update_matching`。

## 7. 攻击实现

### 7.1 创建与能力判定

[attacks/factory.py](../attacks/factory.py) 通过显式 `if/elif` 创建攻击器；[attacks/registry.py](../attacks/registry.py) 是方法状态的单一事实来源。创建后，`supports()` 先判断方法是否实现、是否适用于观测类型以及是否包含所需参数。

当前可运行方法：

| 方法 | 核心实现 |
|---|---|
| `dlg_adapted` | 全局平方 L2 更新匹配，LBFGS 优化 |
| `ig_adapted` | 全局 cosine 匹配 + TV；图像梯度取 sign，Adam 优化 |
| `april_adapted` | L2 + 视觉位置参数梯度 cosine；只适用于满足结构条件的单步梯度 |
| `gradvit_adapted` | 分层 L2 + patch/TV + 后半程外部 BN 先验 |
| `gi_dqa_adapted` | 分层 MSE/cosine + 退火的文档图像先验，不使用表单模板 |
| `random` | 不读取更新的随机负对照 |
| `prior_only` | 只使用图像/公开文本先验，不匹配观测更新 |

闭式 `april`、`idlg`、`dager`、`embedding_recovery`、`decepticons`、`imprint` 和 `mmgia` 目前不会执行真正攻击。

### 7.2 候选变量

[attacks/optim/engine.py](../attacks/optim/engine.py) 中的 `Candidate` 总是创建候选图像；它只为未知文本创建 `[N,L,V]` logits 参数。已知文本使用 observation 中精确公开的 token IDs 构造 one-hot buffer，不参与优化。

候选文本会：

- 屏蔽除 EOS 之外的特殊 token；
- 固定最后一个位置为 EOS，作为公开最大长度边界；
- 用连续 EOS survival 处理软分布；
- 在离散评分和最终输出时取 argmax 并解码。

### 7.3 优化循环

每次优化大致执行：

```text
候选图像/软文本
    -> VLM loss
    -> simulate_update(..., differentiable=True)
    -> 只保留实际上传参数
    -> 与 Observation.tensors 计算匹配目标
    -> 加入方法先验
    -> 对候选变量求导并更新
```

未知图像和未知文本通常交替更新；TAG 文本阶段使用 L2 + 0.01 L1 更新匹配。选择最佳迭代/重启时，统一使用离散候选的归一化更新残差和可选公开语言模型先验，不读取 reference 指标。LAMP 会周期性尝试 token 交换/移动，并用同一公开评分决定是否接受。

预算同时限制 wall time 和 `max_evaluations`。成本输出区分优化迭代、候选更新重放、本地 backward 数、先验调用数、restart 和峰值 CUDA 显存。

攻击 checkpoint 保存候选、优化器、CPU/CUDA RNG、最佳结果、历史和预算计数。恢复时会比较攻击配置、源码 fingerprint、模型、公开更新和公开文本，任一变化都会拒绝续跑。

## 8. 评估实现

[evaluation/reconstruction.py](../evaluation/reconstruction.py) 在 `result.json` 已落盘后读取 `capture/private/`。它先验证 `observation_id`，再对 batch 做一个联合 Hungarian assignment：

```text
pair_cost = image_MSE + normalized_private_text_edit_distance
```

已知文本不会加入 private text cost，也不会计算为恢复成绩。得到配对后，每个样本计算：

- 图像：MSE、PSNR、SSIM；
- 文本：exact match、normalized exact match、WER、word recall、ROUGE-1/2/L；
- 可选视觉/语义：LPIPS、CLIP 图图相似度，以及 caption 的三种 CLIPScore；
- canary：声明数量、实际进入训练 token 的数量、完整实体 exact-match recall；
- 成本：wall time、更新/先验评估次数、restart、迭代和 CUDA 峰值显存。

`token_set_f1()` 已实现为独立 helper，但攻击结果只保存解码文本，没有保存预测 token IDs，因此尚未接入 report。registered-PSNR、CW-SSIM、IIP 和 VLM judge 也尚未实现。

`summarize()` 按实验 condition 聚合报告，保留非完成状态，并使用 image 分组 bootstrap。`evaluation/compare.py` 可以计算攻击相对同一 observation、evaluation config 和 seed 的 `prior_only` 指标差值。

正常任务能力由 `evaluation/utility.py` 评估：VQA 使用项目中注明的轻量 normalization/consensus，caption 使用 `pycocoevalcap` CIDEr；它们不能自动等同于官方 VQA server 或所有论文的标准设置。

## 9. 产物目录与格式

一个完整 Hydra 实验通常生成：

```text
output_dir/
  config.resolved.json         完整 Hydra 配置
  experiment.json              run 索引、状态和结果文件哈希
  data/                        synthetic 时自动生成
    samples.jsonl
    samples.meta.json
    images/
  model/                       rounds=0 时共享的初始模型
    model.json
    model.safetensors
  federation/                  rounds>0 时的联邦训练状态/快照
  run-00000/
    protocol.json              本 run 的严格协议
    capture/
      public/
        observation.json       模型/训练协议、公开文本、参数名及哈希
        update.safetensors      攻击者可见的命名更新
        model_ref.json          指向共享模型快照
        upload.json             defense 与攻击策略说明
      private/
        truth.json              样本、训练后实际文本、canary
        images.safetensors      真实图像
        capture.json            私有数据与环境来源
    attack/
      result.json               Reconstruction（图像 tensor 单独保存）
      images.safetensors        重构图像 batch
      image-000.png             便于人工检查的图像
      evaluation.json           逐样本指标及匹配信息
      checkpoint.json           最新 checkpoint 指针
      checkpoints/              不可变攻击状态 generation
```

`observation.json` 是严格 allowlist，主要包含：

```json
{
  "schema_version": 1,
  "model": {},
  "training": {},
  "model_fingerprint": "...",
  "public_questions": [],
  "public_targets": [],
  "public_question_ids": [],
  "public_target_ids": [],
  "parameter_names": ["..."],
  "update_sha256": "...",
  "observation_id": "..."
}
```

`result.json` 不内嵌图像 tensor，主要字段为：

```json
{
  "status": "completed",
  "reason": "iterations_completed",
  "questions": ["..."],
  "targets": ["..."],
  "costs": {},
  "history": [],
  "provenance": {},
  "observation_id": "...",
  "run_signature": "...",
  "condition": {},
  "environment": {}
}
```

`completed` 只表示生成了可评估的重构，不表示成功泄漏隐私。正式结果需要结合 reconstruction metrics、对照组、失败率和资源成本判断。

## 10. 恢复与可复现性

`ExperimentRunner` 使用 `experiment.json` 管理稳定 run ID。`num_runs` 是停止 ID 的上界，`start_run_id` 是第一个运行 ID。例如 `start_run_id=3 num_runs=5` 执行 run 3 和 4。

恢复实验时需要：

```bash
python examples/run_attack.py \
  start_run_id=3 num_runs=5 resume=true output_dir=outputs/original_run \
  <与原实验相同的协议 override>
```

恢复签名覆盖解析后的配置、数据 manifest SHA-256、模型快照哈希和 Python 源码 fingerprint。输出目录和 run 范围不属于研究协议，可以变化；协议、数据、模型或源码变化会拒绝恢复。

run 按 ID 串行执行。任意异常（包括 OOM）都会先记录当前 run 的 `error` 状态，再立即抛出并终止整个 experiment，不做自动重试。完成 run 的 `result.json` 和 `evaluation.json` 会再次校验哈希后才跳过。

## 11. 测试结构与推荐流程

测试按关注点分为：

| 文件 | 覆盖内容 |
|---|---|
| `test_protocol.py` | SGD/delta 等价、FedAvg、隐私 allowlist、知识条件、二阶导、恢复、上传掩码 |
| `test_hf_adapters.py` | 小型真实 HF LLaVA/BLIP-2/Qwen2.5-VL 架构、LoRA、多步重放、Qwen patchification |
| `test_data_metrics.py` | 数据分组、医疗数据、canary、文本/图像指标、Hungarian 配对、bootstrap |
| `test_priors.py` | 图像先验可微性和 GradViT 外部 BN 先验路径 |
| `test_structure.py` | Hydra 组合、失败落盘与停止、factory/registry、防御、调度器、prior-only 对照 |
| `test_workflows.py` | train/capture/utility、suite 完整性和资源缺失状态 |

推荐从快到慢执行：

```bash
PY=/tmp/giavlm-venv/bin/python

# 1. 查看最终配置，不运行模型
$PY examples/run_attack.py --cfg job --resolve

# 2. 核心数学和隐私协议
$PY -m pytest tests/test_protocol.py -q

# 3. 数据、指标和编排
$PY -m pytest tests/test_data_metrics.py tests/test_structure.py -q

# 4. 全部单元/集成测试
$PY -m pytest -q

# 5. tiny 端到端单条件
$PY examples/run_attack.py attack.iterations=2 attack.checkpoint_interval=1 \
  output_dir=outputs/tiny_e2e

# 6. 多种训练/观测组合的 subprocess smoke
$PY -m core.commands smoke --output outputs/tiny_smoke

# 7. 真实模型前先做 doctor probe
$PY -m core.commands doctor --config configs/qwen2_5_vl.yaml \
  --probe --output outputs/qwen_probe.json
```

真实模型正式运行前还需验证 CUDA、模型权重、CLIP/LPIPS、语言/BN 先验、显存预算和数据许可。`test_hf_adapters.py` 使用随机初始化的小型架构，它证明接口和二阶导链路成立，不证明 3B/7B 预训练模型可以在当前 GPU 上完成攻击。

截至本文整理时，使用 `/tmp/giavlm-venv` 执行全量测试得到 77 项通过、2 项失败；失败原因是新增医疗数据测试需要 `datasets`，但该环境未安装，而且当前 `pyproject.toml` 尚未声明该依赖。因此医疗数据入口应视为代码已实现但环境依赖尚未闭合。该结果应在依赖修复后重新确认。

## 12. 阅读代码的推荐顺序

如果要系统理解实现，建议按执行依赖而不是文件夹字母顺序阅读：

1. `configs/config.yaml` 和一个 tiny 配置；
2. `examples/run_attack.py`、`core/experiment.py`；
3. `core/config.py`、`core/types.py`、`core/knowledge.py`；
4. `core/data.py`、`core/vlm_wrapper.py`、`core/adapters/tiny_llava.py`；
5. `core/fl.py` 与 `tests/test_protocol.py`；
6. `attacks/registry.py`、`attacks/factory.py`、`attacks/optim/engine.py`；
7. `attacks/objectives.py` 和一个方法文件，建议先读 `inverting_gradients.py`；
8. `evaluation/reconstruction.py` 与 `metrics/`；
9. `core/commands.py`、`utils/run_cmds.py` 和 `utils/suite.py`；
10. 最后读 HF adapters 和真实模型配置。

每读一层，都用相应测试验证自己的理解。发现“实现与预期不符”时，应记录：研究预期、源码位置、最小验证实验、实际行为、它属于 bug 还是协议选择，以及修复后需要补充的测试。

## 13. 当前最需要审查的研究选择

以下内容目前是明确的实现选择，后续应优先确认是否符合论文目标：

- 固定块 prompt/response 协议，而不是各 VLM 原生 chat template；
- response-only loss、eval mode、dropout=0；
- LoRA 只覆盖语言 attention projection；
- 本地训练仅实现普通 SGD，无 optimizer state；
- 一个 local step 对应 Batch 中一个不同的连续样本片段；
- capture 对 image 去重，同图多问题不会同时进入一个观测；
- `fed=fedavg` 预设同时选择 FedAvg 服务器规则和单客户端多步 delta 观测，后者不是安全聚合更新；
- 防御后的攻击仍使用 defense-unaware 原始更新重放；
- 多样本匹配成本混合了图像 MSE 和私有文本 edit distance；
- VQA-RAD/SLAKE 只有单参考答案时，utility 退化为 exact match；
- canary 只评估实际通过 tokenizer 和截断进入训练文本的实体；
- 公共数据可能已进入模型预训练，需要 random、prior-only 和 wrong-update 对照才能归因于梯度泄漏。

这些选择不一定是错误，但每一项都会限制实验结论可以推广到什么范围。
