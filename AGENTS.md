# Repository Instructions

## Active Entry and Ownership

- Canonical experiment entry: `python examples/run_attack.py`, backed by
  `core.experiment.main` and real Hydra configuration groups in `configs/`.
- `core/` owns protocol dataclasses, public observations, model adapters,
  client updates, data preparation, artifacts, and experiment lifecycle.
- `attacks/optim/engine.py` owns optimization, budget accounting and checkpoint
  recovery. Method modules own objective/prior combinations. Use explicit
  registration in `attacks/factory.py`; never alias original paper names to
  `*_adapted` methods.
- `defenses/` owns upload transforms. `metrics/` owns metric implementations;
  `evaluation/` alone joins attack outputs to private reference artifacts.
- There is no compatibility import layer. Every module is imported from its
  owning top-level package; the staged CLI runs as `python -m core.commands`.
- `utils/run_cmds.py` runs explicit Hydra jobs from `run_yaml/`, sequentially,
  with argv arrays and no shell execution or GPU occupancy jobs.

## Privacy and Research Contracts

- Never pass reference images, private text, image/sample IDs, data paths,
  private sequence lengths or reference loss masks to an attacker.
- `AdversaryKnowledge` is an explicit assumption, validated against the public
  `Observation`. Only `question_known`/`text_known` authorize exact public text.
- `BaseAttacker.attack` returns `Reconstruction`, not ground truth. Evaluation
  reads truth only after reconstruction is committed. Python type boundaries
  are not an OS sandbox; use a separate user/container for stronger isolation.
- An unimplemented attack degrades to `Reconstruction("not_implemented")` so a
  sweep records the empty cell; an unimplemented defense raises instead, because
  a defense is part of the condition and continuing would mislabel the row.
  `attacks/registry.py` is the single source of truth for the method surface and
  is asserted to cover every name `attacks/factory.py` accepts.
- Report unsupported or unimplemented conditions honestly. Closed-form APRIL,
  iDLG, DAGER, H3 embedding recovery and active-server attacks are not implemented.
- Uploaded defenses currently use defense-unaware raw-update matching; this
  must remain explicit in the output condition. Gaussian perturbation is not
  a certified DP implementation: no privacy accountant or epsilon is provided.
- Runs execute serially. On any exception, save the failed run state and re-raise
  immediately so the experiment stops; do not automatically retry OOM failures or
  alter batch size, update mode, local steps, dtype, task, or knowledge.
- Preserve deterministic image-group splits and shared initial LoRA bases.
  Numerical secure aggregation is not cryptography, nor aggregate inversion.
- Select candidates using observable update/prior scores, never reference metrics.
- Use safetensors and JSON for artifacts; do not introduce pickle checkpoints.

## Configuration and Recovery

- Runtime is Hydra `DictConfig`; `core.config.Config` is the strict serialized
  protocol schema. Translate only through `protocol_config`.
- Groups: `data`, `model`, `attack`, `defense`, `fed`, `knowledge`, `evaluation`.
  Model `name` is a path-safe label; `checkpoint` is the actual model identifier.
- Top-level presets use `<data>_<model>_<attack>_<knowledge>.yaml`.
- `num_runs` is the exclusive stop run ID, not number of additional runs.
  `start_run_id` selects the first requested batch. `resume=true` requires the
  same output directory and matching source/protocol/data fingerprints.
- Preserve unrelated artifacts in `outputs/`, `runs/`, and private datasets.
  Do not share `capture/private/` or resolved private dataset paths in reports.

## Validation

```bash
python -m pytest -q
ruff check core attacks defenses metrics evaluation utils examples tests
python examples/run_attack.py --cfg job --resolve
python examples/run_attack.py attack.iterations=2 attack.checkpoint_interval=1
python -m utils.run_cmds --cmd-config-yaml run_yaml/tiny_smoke.yaml
```

The checkout's validated CPU environment is `/tmp/giavlm-venv`; system Python
does not have the required stack. Do not repair or replace unrelated Conda
environments. Real pretrained GPU experiments require a working CUDA driver,
pinned cached model weights, and optional pretrained metric/prior weights.

The legacy `GI-DQA-Gradient-Inversion-of-Multimodal-Models/` and neighboring
`llm_privacy_eval` repository are references, not modules to edit for this project.
