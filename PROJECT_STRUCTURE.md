# Project Structure

The repository root remains `giavlm/`; it implements the proposed
`fedvlm_gia_bench` layout without adding a redundant nested repository.

```text
giavlm/
  AGENTS.md                   Contributor and privacy-boundary instructions
  README.md                   Canonical commands and research limitations
  PROJECT_STRUCTURE.md        This ownership map
  pyproject.toml / uv.lock     Pinned dependencies and installable packages
  requirements.txt            Editable install backed by pyproject.toml
  install.sh                  Default locked install or --cpu
  core/
    experiment.py             ExperimentRunner, run_experiment, @hydra.main
    vlm_wrapper.py            Abstract VLMAdapter, model factory
    adapters/
      hf.py                   Shared HF loading and fixed-block causal protocol
      llava.py                LLaVA visual feature adapter
      blip2.py                BLIP-2 OPT/Q-Former adapter
      qwen_vl.py              Qwen2.5-VL adapter, not original Qwen-VL
      tiny_llava.py           Offline miniature fixture, not pretrained TinyLLaVA
    fl.py                     FedSGD/FedAvg, model/observation persistence,
                              named accumulation, mask/aggregate utilities
    knowledge.py              AdversaryKnowledge and disclosure validation
    data.py                   COCO/VQAv2, synthetic data, image-group partitions
    config.py                 Strict artifact/wire dataclasses and validation
    types.py                  Batch, Observation, Reconstruction, Support
    artifacts.py              Atomic JSON/safetensors and fingerprints
    commands.py               Staged data/train/capture/attack/evaluate CLI
  attacks/
    base.py / factory.py       Public-only attacker API and explicit registration
    optim/
      engine.py               Shared differentiable replay, budgets, recovery
      dlg.py                  DLG adaptation objective and attacker
      inverting_gradients.py  Cosine + TV adaptation
      gradvit.py              Layer objective + BN/patch priors
      gidqa.py                Template-free document prior adaptation
      idlg.py                 Explicit not_implemented boundary
    analytic/                 april, dager, embedding_recovery capability stubs
    text/                     TAG matching and LAMP-style discrete proposals
    malicious/                decepticons, imprint capability stubs
    priors.py                 Existing image priors and public text/BN priors
    objectives.py             Differentiable named-update matching
    registry.py               Existing adaptation support checks
    prior_only.py             No-update control
  defenses/                   base/factory, none, clipping, gaussian_dp,
                              topk_sparsify, sign_sgd; pre-training stubs
  metrics/
    image.py                  MSE, PSNR, SSIM
    text.py                   ROUGE/EM/WER/word recall; token/canary helpers
    semantic.py               Optional LPIPS and CLIP evaluator loading/scoring
    cost.py                   Wall time, evaluations and per-GPU peak memory
  configs/
    config.yaml               Hydra defaults and run/sweep output routing
    data/ model/ attack/ defense/ fed/ knowledge/ evaluation/
    <data>_<model>_<attack>_<knowledge>.yaml
    tiny.yaml, llava.yaml, ... Legacy flat configs for the staged CLI only
  examples/run_attack.py      Canonical experiment entry
  run_yaml/tiny_smoke.yaml    Example declarative job schedule
  utils/
    run_cmds.py               Dry-run by default, opt-in sequential execution
    suite.py                  Legacy shared-model multi-condition suite
  evaluation/
    reconstruction.py        Joint matching, reference evaluation, bootstrap
    utility.py               Held-out VQA/caption utility
    compare.py               Paired metric deltas versus prior_only
  jupyter/                    Analysis notebook workspace (no fabricated results)
  figs/                       Figure workspace
  tests/                      Legacy regression + new Hydra/protocol tests
  docs/                       Detailed protocol/baselines/validation notes
  GI-DQA-Gradient-Inversion-of-Multimodal-Models/  Untouched reference
```

## Data Flow

Hydra groups -> strict protocol + adversary knowledge -> prepare manifest ->
shared initial model or resumable federation -> per-run client update ->
upload defense -> public observation -> attacker -> committed reconstruction ->
private evaluation -> per-run result and experiment state.

`fed=fedavg` means observing a client's multi-step delta. It does not mean
observing a secure sum of several clients. `fed.rounds>0` additionally trains
a federation before capture; `model_snapshot=...` selects an existing checkpoint
instead. These two options cannot be combined.

Each run owns `capture/public/`, `capture/private/`, and `attack/`. Shared model
weights live once at experiment level. `experiment.json` indexes stable run IDs;
completed result/evaluation hashes are checked before skipping a run. Attack
checkpoints include optimizer state, candidate state, RNG and budget counters.

## Implemented Versus Planned

- Runnable: existing VLM adaptations, both update modes, full/LLM-full/LoRA,
  private/question-known/text-known, five post-update defenses, Hydra repeats,
  resume, bounded OOM recovery, and the legacy staged workflows.
- Helpers only: named full/LoRA aggregation, parameter upload masks, numerical
  secure-aggregation mean, tokenizer-ID set F1, canary entity exact-match recall.
  Masks/aggregate observations fail closed in the experiment entry until replay
  and attribution are implemented. Token/canary helpers are not automatically
  included in existing reconstruction reports.
- `data=medical_vqa` accepts the existing normalized JSONL VQA schema only. It
  does not download/convert a medical corpus or inject synthetic PII. Such
  dataset-specific preparation and canary loss protocols remain future work.
- Not implemented: original closed-form APRIL, iDLG, DAGER, H3 embedding recovery,
  malicious-server attacks, token obfuscation, safe-template training defense,
  registered-PSNR, CW-SSIM, IIP and VLM-judge answerability.
- Existing differentiable priors/objectives were relocated; this change does
  not claim a complete port of the proposed `breaching` classes or metrics.
  The legacy GI-DQA source (including its bugs) is intentionally not modified.

The new structure follows the active Hydra/runner/factory conventions of
`../llm_privacy_eval`, not its obsolete structure document. Name-based LoRA
alignment and run-ID recovery follow that project's patterns; full-tuning
support, fail-before-mutation aggregation, and no-reference attacker contracts
are benchmark-specific adjustments.
