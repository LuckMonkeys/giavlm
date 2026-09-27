# Repository Instructions

FedVLM gradient-inversion benchmark. Entry: `python examples/run_attack.py`
(`core.experiment.main`, Hydra groups in `configs/`); staged CLI:
`python -m core.commands`. Read `docs/PROJECT_HANDOFF.md` next for current state.

Docs: `docs/protocol.md` (threat model, artifacts, boundaries) ·
`docs/project_architecture.md` (Chinese code walkthrough) · `docs/validation.md` ·
`docs/baselines.md` (paper fidelity). The legacy
`GI-DQA-Gradient-Inversion-of-Multimodal-Models/` and `llm_privacy_eval` are
read-only references.

## Ownership

- `core/`: protocol dataclasses, observations, adapters, client updates, data,
  artifacts, lifecycle. No compatibility import layer; import from owning package.
- `core/aggregation.py`: federated algorithms. Keep client/upload/aggregate/apply
  stages explicit; register in `create_federated_algorithm`. A stateful server
  rule needs JSON/safetensors checkpoint recovery first.
- `attacks/optim/engine.py`: optimization, budgets, checkpoints. Method modules own
  objective/prior combinations, registered explicitly in `attacks/factory.py`.
  `attacks/registry.py` is the single source of truth for the method surface.
  Never alias original paper names to `*_adapted` methods.
- `attacks/init.py`: candidate starts (random / `public_image` / `private_reference`
  images and text) and their perturbations; the engine applies them per restart.
- `defenses/`: upload transforms. `metrics/`: metrics. Only `evaluation/` joins
  attack outputs with private references, including the private-reference
  diagnostics in `evaluation/gradient_diagnostics.py` (`diagnose-gradient`).
- `utils/run_cmds.py`: argv-only job runner for `run_yaml/`; never add shell
  execution. `run.sh` is a notebook of copyable one-liners, not a script.

## Hard Rules

- Never give an attacker private images, private text, sample/image IDs, data
  paths, or reference loss masks. A field is public only when the knowledge
  condition declares it (`question_known`, `text_known`, `image_known`,
  `image_question_known`), and it reaches the attacker only through `Observation`.
  Per-sample token lengths require `token_lengths_known`. `init_source` /
  `init_text_source=private_reference` starts from the private reference; it is a
  labeled diagnostic, never a benchmark result.
- Attacks return `Reconstruction`, never truth; evaluation reads truth only after
  the reconstruction is committed. Select candidates by observable update/prior
  scores, never by reference metrics.
- Unimplemented attack → `Reconstruction("not_implemented")`; unimplemented
  defense → raise. Report unsupported conditions honestly and never overclaim
  (see `protocol.md#Experimental Boundaries`: no certified DP, no crypto secure
  aggregation, defense-unaware matching must stay visible in the condition).
  Not implemented: closed-form APRIL, iDLG, DAGER, H3 embedding recovery,
  active-server attacks.
- Fail fast: on any exception save failed-run state and re-raise. Never retry OOM
  or silently change batch size, update mode, local steps, dtype, task, knowledge.
- Keep deterministic image-group splits and shared initial LoRA bases.
- Artifacts are safetensors + JSON only; no pickle.
- Preserve unrelated files in `outputs/`, `runs/`, private datasets, and
  uncommitted work. Never share `capture/private/` or resolved private paths.

## Configuration

- Hydra `DictConfig` enters strict `core.config.Config` only via `protocol_config`.
- Groups: `data`, `model`, `attack`, `defense`, `fed` (algorithm + client
  optimizer), `tuning` (F-C/F-L/F-CL/F-2stage), `knowledge`, `evaluation`.
  Model `name` is a path label; `checkpoint` is the real identifier.
- Presets: `<data>_<model>_<attack>_<knowledge>.yaml`.
- Knowledge conditions are sets of public fields (`KNOWLEDGE_FIELDS` in
  `core/config.py`); public fields are fixed inputs, never optimized or scored.
- `attack.image_dtype` (default float32) is the candidate-image master copy,
  independent of the victim `model.dtype`; bf16 candidates round away small steps.
- Attack starts are declared conditions: `attack.init_*` / `attack.init_text_*` are
  recorded in the result condition; see `protocol.md#Attack Initialization`.
- `num_runs` is the exclusive stop run ID; `start_run_id` is the first. `resume=true`
  needs the same output dir and matching source/protocol/data fingerprints.
- Model overlays are schema v4 and `Observation` is schema v5 (adds declared public
  images): reject older schemas, never migrate silently. Attack checkpoints version
  their schema independently.

## Environment and Validation

- GPU experiments: Conda env `gia` (CUDA, cached pinned weights).
- CPU tests: `/tmp/giavlm-venv`. System Python lacks the stack. Do not repair or
  replace unrelated Conda environments.
- Long GPU runs: launch detached (`setsid nohup … & disown`; a plain `&` inside a
  tool call dies with it) and arm a watcher that exits on completion, failure, or a
  dead scheduler (`outputs/diagnostics/analysis_scripts/watch_*.sh`). Verify the
  first capture/checkpoint before leaving a run unattended.
- The login shell is zsh, which does not word-split `$VAR`; put multi-argument
  loops in a bash script. `run_yaml/` generic-job `args` must be quoted strings and
  job names match `[a-zA-Z0-9_-]+` (no dots).

```bash
python -m pytest -q
ruff check core attacks defenses metrics evaluation utils examples tests
python examples/run_attack.py --cfg job --resolve
python examples/run_attack.py attack.iterations=2 attack.checkpoint_interval=1
python -m utils.run_cmds --cmd-config-yaml run_yaml/tiny_smoke.yaml
```

## Session Start

Before acting, check `git status`, recent commits, the process table, and GPU
state. A scheduler row marked `running` does not prove the process is alive.
Update `docs/PROJECT_HANDOFF.md` after major commits, experiment phases, or
direction changes; keep transient state there, not here.
