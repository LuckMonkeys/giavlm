# Validation

## Automated Checks

Run `pytest -q` for protocol, data, metrics and model-adapter tests. The offline
suite requires no pretrained weights or network access. It checks:

- Single-step delta/gradient equivalence for full, LLM-full and LoRA modes.
- Functional multistep replay against actual `torch.optim.SGD`.
- Weighted FedAvg from a common initial state and LoRA-only parameter exposure.
- Exact observation replay, wrong-input residual and second-order derivatives
  with respect to both image and private text candidates.
- Standard LoRA first-step zero A gradients/nonzero B gradients.
- Public text exclusion from optimized variables, observation allowlisting,
  metadata integrity and prevention of reference-text export.
- EOS, repeated-token metrics, complete-pair Hungarian matching, image-group
  bootstrap and preservation of non-completed status counts.
- Candidate/optimizer checkpoint resume and finite attack results.
- Actual small Transformers LLaVA, BLIP-2 and Qwen2.5-VL architectures in full and
  LoRA modes, multistep LoRA replay, and Qwen patchification parity with the official processor.
- External GradViT BN and patch priors preserve image gradients; the second-half
  prior schedule is exercised with explicitly synthetic test-only weights.
- Training/checkpoint/capture/utility workflows, sealed suite inputs, and actual
  interrupted optimization resume compared against a continuous run.

`python -m core.commands smoke` additionally executes prepare/capture/attack/evaluate/report in
separate Python processes across eight task/training/upload combinations.
These synthetic runs test execution and contracts, not pretrained-model privacy.

## Hydra Layout Validation (2026-09-08)

- All original regression tests remain enabled after moving implementation out
  of the former `src/giavlm/` layer, which has since been removed entirely;
  new tests cover Hydra groups/presets, run-ID resume, protocol
  mismatch rejection, shared model snapshots, pre-capture federated training,
  bounded OOM recovery, named aggregation, defense metadata and paired controls.
- Executed both jobs in `run_yaml/tiny_smoke.yaml`: full/FedSGD/private and
  LoRA/FedAvg/private. Both completed reconstruction and evaluation. Resuming a
  completed Hydra experiment preserved its result artifacts.
- Re-executed all eight legacy subprocess smoke cases after migration: both
  tasks, full/LoRA, and gradients/client deltas. All eight reports completed.
- Built an offline wheel and checked that it includes implementation, canonical
  entry and Hydra YAML data groups. The installed console entry resolves configs
  from outside the checkout. `uv lock --check --offline` and shell syntax checks
  for `install.sh` pass.
- CPU validation artifacts are in `/tmp/giavlm-layout-validation` and
  `/tmp/giavlm-layout-legacy-smoke`; they contain synthetic fixtures only.

OOM tests inject `torch.OutOfMemoryError`; they do not demonstrate recovery from
a physical GPU allocation failure. The existing GPU/pretrained limitations below
remain unchanged. Reorganization changes the source fingerprint, so pre-migration
checkpoints are preserved but cannot be resumed under a different implementation.

## Before Formal Results

1. Cache the pinned victim, text-prior, CLIP and LPIPS weights in writable paths.
2. Supply the GradViT prior and record its training source and hash.
3. Run `doctor --probe` for each exact model, precision, training mode and device
   placement. The test constructs second derivatives, not just a forward pass.
4. Run 4 samples for correctness, then 16 tuning samples for cost profiling and
   independent attack hyperparameter selection. Freeze the resulting configs.
5. Materialize the 100-sample/3-seed matrix and inspect `budget.json` before
   execution. Compare privacy together with utility, costs and failure counts.

## Limits of This Checkout's Validation

The execution host has no usable CUDA driver. No pretrained 7B weights, full-scale
COCO/VQAv2 experiment, external language/image prior or LPIPS/CLIP weight-based
score has been validated here. The multi-GPU placement path needs GPU validation.
This checkout does not establish baseline ranking, attack success rates, model
privacy, or comparable downstream utility at the supplied default learning rates.

The core provides PSNR/SSIM/ROUGE/edit metrics without downloading weights. Requested
optional metrics fail if dependencies/weights are unavailable, rather than writing
fabricated values. VQA utility's simplified normalization is identified in its
output and should not be quoted as the official VQA server score. Caption utility
uses the reference pycocoevalcap implementation when installed with Java.
