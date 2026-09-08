# GIAVLM: FedVLM GIA Benchmark

A protocol-first benchmark for reconstructing private images and text from
federated vision-language model updates. The original GI-DQA project remains in
`GI-DQA-Gradient-Inversion-of-Multimodal-Models/`; the new package does not import it.

**Status:** runnable research infrastructure, with CPU correctness tests on the
offline fixture and small Transformers LLaVA, BLIP-2 and Qwen2.5-VL architectures.
Pretrained 7B experiments have not been validated on GPUs in this workspace.
All transferred attacks are explicitly named `*_adapted`; these are not claims
of numerical reproduction of the original papers.

The project now follows the Hydra organization of `llm_privacy_eval`.
Implementation lives in `core/`, `attacks/`, `defenses/`, `metrics/`, and
`evaluation/`. There is no compatibility import layer.
See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for ownership and the explicit
implemented/planned feature boundary, and [AGENTS.md](AGENTS.md) for contributor rules.

## Install

Python 3.10+ is required. Direct dependencies are pinned in `pyproject.toml` and
the complete resolver output is in `uv.lock`.

`bash install.sh` runs the locked install below; `bash install.sh --cpu` creates
an isolated `.venv` with CPU PyTorch. `requirements.txt` delegates dependency
versions to project metadata instead of maintaining a second pin list.

```bash
uv sync --frozen --extra dev --extra metrics
uv run python -m core.commands doctor
uv run pytest -q
```

For a CPU-only environment, install PyTorch first from its CPU index:

```bash
uv venv .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
uv pip install --python .venv/bin/python -e '.[dev,metrics]'
.venv/bin/python -m core.commands smoke --output runs/smoke
```

The development validation environment for this checkout is
`/tmp/giavlm-venv/bin/python`, which matches the pins above exactly. It lives
under `/tmp` and does not survive a reboot; recreate it with the commands above
in a durable path before relying on it. The Conda environment `gia`
(torch 2.6.0+cu124, transformers 4.49.0, peft 0.14.0) also runs the full suite
and is the one to use once GPUs are available.
`smoke` runs each stage in a separate process and covers both tasks, full/LoRA,
and gradient/multistep-delta observations. Use `--quick` for one pipeline.

## Run One Experiment

The canonical entry is `examples/run_attack.py` (installed alias: `fedvlm-gia`).
It composes Hydra groups, prepares the offline fixture, captures updates, runs
the attack, and evaluates committed output:

```bash
python examples/run_attack.py --cfg job --resolve
python examples/run_attack.py attack.iterations=20 output_dir=outputs/toy
python examples/run_attack.py fed=fedavg fed.mode=lora_llm knowledge=question_known defense=clipping
python examples/run_attack.py --config-name vqav2_llava_ig_private data.manifest=/datasets/prepared/samples.jsonl
```

The last command requires cached pinned model/prior/metric weights and a working
CUDA stack. `model=qwen_vl` selects Qwen2.5-VL, not the original Qwen-VL model.
`tiny_llava` is a miniature offline correctness fixture, not pretrained TinyLLaVA.

Run IDs select distinct client batches. `num_runs` is the exclusive stop ID,
not a count of extra runs. Resume requires the same output directory and exact
protocol/source/data; batch size and dtype never change automatically after OOM:

```bash
python examples/run_attack.py num_runs=3 output_dir=outputs/repeated
python examples/run_attack.py num_runs=5 start_run_id=3 resume=true output_dir=outputs/repeated
python -m utils.run_cmds --cmd-config-yaml run_yaml/tiny_smoke.yaml
python -m utils.run_cmds --cmd-config-yaml run_yaml/tiny_smoke.yaml --execute
```

Hydra also supports `-m attack=ig_adapted,dlg_adapted knowledge=private,text_known`.
Use Hydra's default per-job output directories for sweeps, not one shared explicit
`output_dir`. The YAML scheduler is sequential and dry-run by default. GPU use is
explicit via `--gpu-ids 0,1`; it has no background occupancy behavior.

`fed.rounds=0` captures the initial model. Set `fed.rounds>0` to train first, or
set `model_snapshot=/path/to/round-0010` to capture from an existing checkpoint.
`defense={none,clipping,gaussian_dp,topk_sparsify,sign_sgd}` transforms the upload;
current attacks are explicitly **defense-unaware**. `gaussian_dp` provides client
clipping plus Gaussian noise, not an epsilon/delta guarantee or privacy accountant.
Aggregate inversion and upload-mask replay reject unsupported configuration
instead of silently running an individual raw-gradient experiment.

`data=medical_vqa` requires a pre-normalized manifest in the existing schema;
canary injection is not implemented. Closed-form
APRIL, iDLG, DAGER, H3 embedding recovery, and malicious-server modules return
`not_implemented`; directory presence is not an implementation claim.

## Legacy Staged Commands

The following API and flat `configs/{tiny,llava,blip2,qwen2_5_vl}.yaml` files
remain available for existing scripts. They are not Hydra presets.

```bash
python -m core.commands prepare-data --synthetic --count 96 --clients 1 --output data/toy
python -m core.commands capture --config configs/tiny.yaml --data data/toy/samples.jsonl --client 0 --output runs/example/capture --set training.clients=1 --set training.clients_per_round=1
python -m core.commands attack --observation runs/example/capture/public --output runs/example/attack
python -m core.commands evaluate --reconstruction runs/example/attack --truth runs/example/capture/private
python -m core.commands report --input runs/example --output runs/example/report.json
```

`capture/public/` contains only the model/update and declared public text.
`capture/private/` contains references and preparation provenance. The attack
command does not accept a dataset or reference path. For a stronger boundary,
run it under an OS user/container that cannot read the private directory; Python
type isolation alone is not a filesystem sandbox.

Override configuration with repeated `--set key=value`. Examples:

```bash
# A client upload after five local minibatches, not an averaged gradient.
python -m core.commands capture --data data/toy/samples.jsonl --output runs/delta/capture --set training.observation=client_delta --set training.local_steps=5
# Resume only an identical attack; budgets/configurations cannot silently change.
python -m core.commands attack --observation runs/example/capture/public --output runs/example/attack --resume
```

## Prepare COCO and VQAv2

Obtain official COCO caption/image files and matching VQAv2 question/answer
annotations. No datasets or model weights are downloaded implicitly by preparation.

```bash
python -m core.commands prepare-data --captions /datasets/coco/annotations/captions_train2014.json --images /datasets/coco/train2014 --questions /datasets/vqa/v2_OpenEnded_mscoco_train2014_questions.json --annotations /datasets/vqa/v2_mscoco_train2014_annotations.json --output data/coco-vqa
```

The splitter groups by `coco:<image_id>` across tasks and annotations. It uses
70% train / 10% attack tuning / 20% attack evaluation buckets, and deterministic
IID client assignments. This is a benchmark-specific split, not the official
task test split. The exact supervised VQA answer is `multiple_choice_answer`;
all annotator answers remain available to the utility evaluator only.

Preprocessing is a deterministic bicubic center crop to a public square size.
Reconstruction metrics concern this actual model input, not unseen cropped-out
pixels. Fixed question/target blocks include a public maximum-length EOS;
private lengths, EOS locations and padding masks are never exported.

## Pretrained Models and Federation

Recipes are `configs/llava.yaml`, `configs/blip2.yaml`, and
`configs/qwen2_5_vl.yaml`. They pin checkpoint hashes, start with LLM-only LoRA,
use float32 weights/updates, and require cached weights. Populate the HF cache separately, or explicitly set
`model.local_files_only=false` for the initial model download. Set a writable
`HF_HOME` and `TORCH_HOME` when running in a restricted container.

```bash
python -m core.commands doctor --config configs/llava.yaml --probe --output runs/llava-probe.json
python -m core.commands train --config configs/llava.yaml --data data/coco-vqa/samples.jsonl --output runs/federation
python -m core.commands capture --config configs/llava.yaml --model runs/federation/round-0010 --data data/coco-vqa/samples.jsonl --client 0 --output runs/round10/capture
python -m core.commands utility --model runs/federation/round-0010 --data data/coco-vqa/samples.jsonl --device cuda:0 --output runs/round10/utility.json
```

Training uses functional SGD, weighted FedAvg, no momentum/weight decay, disabled
dropout, and the versioned `fixed-block-eos-v1` format. It stores rounds 0/10/20
and two rotating recovery checkpoints. `train --resume` restores model state
and deterministic round sampling. Caption runs should set `training.task=caption`
and `model.target_length=64` on real models. Set `training.observation=client_delta`
with `training.local_steps=5` to train/capture five local steps.

`training.mode=full` trains every model parameter; `llm_full` trains the language
model only; `lora_llm` trains Q/K/V/O LoRA A/B parameters only. LoRA defaults to
rank 8, alpha 16, zero dropout and PEFT's standard initialization. Every client
starts from the same global A/B state; updates and aggregation stay in A/B space.

Independent experiments can run on separate GPUs. Experimental within-model
placement uses `model.device_map=balanced` and
`model.max_memory={0: '70GiB', 1: '70GiB'}`. CPU/disk offload is rejected. Run the
second-order `doctor --probe` on the exact hardware/configuration before using
this path: multi-GPU dispatch has not been verified here. No quantization or
FlashAttention path is silently substituted after OOM.

The default float32 protocol avoids mistaking low-precision update rounding for
privacy protection. An explicit `model.dtype=bfloat16` uses literal bfloat16
weights and SGD updates, not a hidden float32 master-weight optimizer. Treat it
as a separate numerical protocol and inspect zero updates and task utility.

## Attacks and Controls

See [baseline notes](docs/baselines.md) for formulas, adaptations and references.
The implemented visual methods are `dlg_adapted`, `ig_adapted`, `april_adapted`,
`gradvit_adapted`, and `gi_dqa_adapted`. Text components are `tag_adapted` and
`lamp_adapted`. The offline default is TAG; real recipes default to LAMP with a
pinned GPT-2 prior. LAMP requires that public prior to be cached, or explicit
`attack.allow_prior_download=true`.

GradViT requires `attack.gradvit_prior_checkpoint=/path/to/resnet50.safetensors`.
Provide a MoCo-v2 ResNet50 backbone checkpoint in torchvision parameter naming;
the checkpoint hash is recorded. The runner never substitutes random BN weights
or a different prior for a missing file. A different training source must be
reported as another adaptation. DAGER/MMGIA are audit entries with explicit
`not_implemented` status; original GI-DQA is `not_applicable` in the template-free
track. Its original code remains available for separate document experiments.

Controls: `attack.method=random`, `attack.method=prior_only`, and
`attack --wrong-observation OTHER_PUBLIC_DIRECTORY`. The wrong-update control
requires the same model and protocol, and an independent update. Public text is
still available in knowledge-conditioned controls. Under `private`, all text
starts from random candidates, with no reference-driven initialization.

## Experiment Suites and Metrics

Materialize a pilot before executing a large matrix:

```bash
python -m core.commands suite --configs configs/llava.yaml --data data/coco-vqa/samples.jsonl --samples 16 --split tune --seeds 0 --methods ig_adapted --output runs/pilot
python -m core.commands run-suite --manifest runs/pilot/jobs.jsonl --limit 1
```

`suite` writes sealed configs, argv-based jobs, and GPU-hour budgets; it does not
run them. `run-suite` shares initial model checkpoints across samples, reuses
identical captures, and resumes attacks. Default
publication settings select 100 samples per condition and three attack seeds.
Use `--pilot-seconds` from actual timing to obtain an estimate in `budget.json`.
Capture/model loading/evaluation costs are additional. Suites without `--model`
run at initialization; trained checkpoints are exercised with the explicit
`train`/`capture --model` workflow above.

Core metrics are PSNR/SSIM, ROUGE-1/2/L, exact match, WER, and word recall.
LPIPS and the three caption CLIPScore directions are opt-in for tiny experiments
and enabled in real-model recipes. Required metric weights must be cached;
missing requested metrics fail evaluation rather than being replaced by proxies.
The `psnr_infinite` flag represents exact identity without invalid JSON Infinity.
Batch alignment uses one joint image/text Hungarian assignment. Paired image/text
alignment agreement is diagnostic, not an unqualified attack success rate.

`utility` reports VQA soft accuracy with documented simplified normalization.
Caption CIDEr uses `pycocoevalcap` plus its PTB tokenizer and requires Java:
install with `uv sync --extra utility`. Attack text scores compare the actual
trained, truncated token sequence, not another semantically valid annotation.

Every result distinguishes `completed`, `attack_failed`, `not_applicable`,
`not_implemented`, and `resource_unavailable`. Completion means a reconstruction
was produced, not that privacy was broken. Reports retain status denominators,
group bootstrap by image, and average repeated annotations/seeds within each
image before computing intervals. Public COCO data may be in pretraining, so
compare no-update and wrong-update controls before attributing results to leakage.

## Further Documentation

- [Protocol and observation contract](docs/protocol.md)
- [Baseline fidelity and applicability](docs/baselines.md)
- [Validation and remaining experimental limits](docs/validation.md)
