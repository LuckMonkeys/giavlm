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

## Real-Model and Real-Data Validation (2026-09-08)

Medical VQA ingestion, run on the real corpora rather than a fixture:

- `prepare-data --medical vqa_rad slake` produced 9277 rows over 956 unique
  images. `read_manifest` accepted the manifest, so no image group crossed a
  split or client boundary, and `assignment` reproduced every row's placement.

Second-order probe on a real pretrained VLM, CPU only:

- `doctor --probe` passed on Qwen2.5-VL-3B-Instruct (3.76B parameters, float32,
  CPU, 112px, `lora_llm`) in 98s. `replay_max_error` was exactly 0.0 and the
  candidate gradient norms were finite and nonzero for images (114.8), questions
  (1.92) and targets (1.89). The `pixel_values -> loss` path is therefore
  differentiable end to end through a current `transformers` VLM; both LLaVA and
  Qwen2-VL insert visual features with `inputs_embeds.masked_scatter`, and the
  two `@torch.no_grad()` sites in that code sit on Qwen's integer-only RoPE
  forward and on BLIP-2's `generate`, neither of which is on the loss path.
- The probe reported 144 of 288 trainable tensors as zero-valued in the upload.
  All 144 are `lora_A`. PEFT initializes B at zero, so at the first federated
  round `grad(A) = (alpha/r) B^T G^T X` vanishes and only
  `grad(B) = (alpha/r) G^T X A^T` is uploaded: input activations reach the server
  solely through the r-dimensional random projection A until B moves off zero.
  `test_lora_a_carries_no_signal_at_initialization` locks this in. Any
  first-round LoRA attack is bounded by it.

Attack cost on the same model, measured not estimated:

- `ig_adapted` on a real VQA-RAD/SLAKE image, CPU, float32, 112px, `lora_llm`:
  10 iterations in 228s, i.e. **22.8s per iteration**. Extrapolating, the
  `iterations: 1000` in `configs/qwen2_5_vl.yaml` is 6.3 hours per sample per
  restart, and 24000 iterations is 152 hours. Ten iterations recovered nothing
  (PSNR 8.26, SSIM 0.007, ROUGE-L 0.0), as expected at that budget.
  **This is a cost probe, not a result.** Optimization attacks at publication
  budgets are GPU work.

Pipeline comparison on real images with the 8px tiny fixture, 4 runs, 60
iterations, `prior_only` control paired per run:

| method | PSNR | SSIM | target ROUGE-L | target EM |
|---|---|---|---|---|
| prior_only | 10.24 | 0.014 | 0.00 | 0.00 |
| ig_adapted | 6.96 | 0.049 | 0.35 | 0.25 |
| dlg_adapted | 7.33 | 0.034 | 0.35 | 0.25 |
| random | 7.09 | -0.025 | 0.00 | 0.00 |

The control outscores every attack on PSNR and MSE, so absolute image fidelity
here reflects the smoothness prior rather than the observed update. The attacks
separate from `random` only on SSIM and on the text metrics. This is a fixture
result about the pipeline and the metrics, not a privacy claim about any real
model.

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

No pretrained 7B weights, full-scale COCO/VQAv2 experiment, external
language/image prior or LPIPS/CLIP weight-based score has been validated here.
Real-model validation reached a 3B checkpoint on CPU at a 10-iteration budget
only; no attack has been run to convergence on any pretrained VLM. The multi-GPU placement path needs GPU validation.
This checkout does not establish baseline ranking, attack success rates, model
privacy, or comparable downstream utility at the supplied default learning rates.

The core provides PSNR/SSIM/ROUGE/edit metrics without downloading weights. Requested
optional metrics fail if dependencies/weights are unavailable, rather than writing
fabricated values. VQA utility's simplified normalization is identified in its
output and should not be quoted as the official VQA server score. Caption utility
uses the reference pycocoevalcap implementation when installed with Java.
