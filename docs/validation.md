# Validation

## Automated Checks

Run `pytest -q` for protocol, data, metrics and model-adapter tests. The offline
suite requires no pretrained weights or network access. It checks:

- Single-step delta/gradient equivalence for F-C, F-L, F-CL and F-2stage.
- Functional multistep replay against actual `torch.optim.SGD`.
- Explicit FedSGD/FedAvg rules, weighted updates from a common initial state,
  fail-before-mutation application and LoRA-only parameter exposure.
- Exact observation replay, wrong-input residual and second-order derivatives
  with respect to both image and private text candidates.
- Standard LoRA first-step zero A gradients/nonzero B gradients.
- Public text exclusion from optimized variables, observation allowlisting,
  metadata integrity and prevention of reference-text export.
- EOS, repeated-token metrics, complete-pair Hungarian matching, image-group
  bootstrap and preservation of non-completed status counts.
- Candidate/optimizer checkpoint resume and finite attack results.
- Actual small Transformers LLaVA, BLIP-2 and Qwen2.5-VL architectures under all
  four strategies, multistep LoRA replay, and Qwen patchification parity with the official processor.
- External GradViT BN and patch priors preserve image gradients; the second-half
  prior schedule is exercised with explicitly synthetic test-only weights.
- Training/checkpoint/capture/utility workflows, sealed suite inputs, and actual
  interrupted optimization resume compared against a continuous run.

`python -m core.commands smoke` additionally executes prepare/capture/attack/evaluate/report in
separate Python processes across 24 task/strategy/upload combinations.
These synthetic runs test execution and contracts, not pretrained-model privacy.

## Hydra Layout Validation (2026-09-08)

- All original regression tests remain enabled after moving implementation out
  of the former `src/giavlm/` layer, which has since been removed entirely;
  new tests cover Hydra groups/presets, run-ID resume, protocol
  mismatch rejection, shared model snapshots, pre-capture federated training,
  fail-fast error persistence, named federated algorithms, defense metadata and paired controls.
- Executed both jobs in `run_yaml/tiny_smoke.yaml`: F-L/FedSGD/private and
  F-CL/FedAvg/private. Both completed reconstruction and evaluation. Resuming a
  completed Hydra experiment preserved its result artifacts.
- The subprocess smoke matrix now covers both tasks, all four tuning strategies,
  and FedSGD/FedAvg-SGD/FedAvg-AdamW uploads.
- Built an offline wheel and checked that it includes implementation, canonical
  entry and Hydra YAML data groups. The installed console entry resolves configs
  from outside the checkout. `uv lock --check --offline` and shell syntax checks
  for `install.sh` pass.
- CPU validation artifacts are in `/tmp/giavlm-layout-validation` and
  `/tmp/giavlm-layout-legacy-smoke`; they contain synthetic fixtures only.

Failure tests inject exceptions, including `torch.OutOfMemoryError`, and verify
that one failed attempt is persisted before the serial experiment stops. The
existing GPU/pretrained limitations below remain unchanged. Reorganization changes
the source fingerprint, so pre-migration checkpoints are preserved but cannot be
resumed under a different implementation.

## Real-Model and Real-Data Validation (2026-09-08)

Medical VQA ingestion, run on the real corpora rather than a fixture:

- `prepare-data --medical vqa_rad slake` produced 9277 rows over 956 unique
  images. `read_manifest` accepted the manifest, so no image group crossed a
  split or client boundary, and `assignment` reproduced every row's placement.

Second-order probe on a real pretrained VLM, CPU only:

- `doctor --probe` passed on Qwen2.5-VL-3B-Instruct (3.76B parameters, float32,
  CPU, 112px, F-L) in 98s. `replay_max_error` was exactly 0.0 and the
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

- `ig_adapted` on a real VQA-RAD/SLAKE image, CPU, float32, 112px, F-L:
  10 iterations in 228s, i.e. **22.8s per iteration**. Extrapolating, the
  `iterations: 1000` in `configs/qwen2_5_vl.yaml` is 6.3 hours per sample per
  restart, and 24000 iterations is 152 hours. Ten iterations recovered nothing
  (PSNR 8.26, SSIM 0.007, ROUGE-L 0.0), as expected at that budget.
  **This is a cost probe, not a result.** Optimization attacks at publication
  budgets are GPU work.

Upload-mask axis on real medical images, tiny fixture, 3 runs, 60 iterations:

| upload | tensors | PSNR | SSIM | target ROUGE-L | question ROUGE-L |
|---|---|---|---|---|---|
| whole LoRA update | 8 | 5.807 | 0.031 | 0.133 | 0.111 |
| `*lora_B*` only | 4 | 5.807 | 0.031 | 0.133 | 0.111 |

The two are identical to every reported digit, and inspecting the uploads shows
why: all four `lora_A` tensors in the unmasked upload are exactly zero, so
withholding them removes nothing. **Uploading only LoRA-B is not a defense at the
first federated round.** Whether it becomes one after B moves off zero is
untested here and needs a round-indexed sweep.

Attack-budget sensitivity on the tiny fixture, 4 runs, `prior_only` and `random`
paired per run. Two budgets, because the first one inverts the conclusion:

At 60 iterations on real medical images, 8px, where the fixture's 27-word
vocabulary truncated two of four targets to the empty string:

| method | PSNR | SSIM | target ROUGE-L | target EM |
|---|---|---|---|---|
| prior_only | 10.24 | 0.014 | 0.00 | 0.00 |
| ig_adapted | 6.96 | 0.049 | 0.35 | 0.25 |
| dlg_adapted | 7.33 | 0.034 | 0.35 | 0.25 |
| random | 7.09 | -0.025 | 0.00 | 0.00 |

At 2000 iterations on synthetic data, 16px, where the text is representable in
the fixture vocabulary:

| method | PSNR | SSIM | target EM | target ROUGE-L | question ROUGE-L |
|---|---|---|---|---|---|
| prior_only | 6.59 | 0.026 | 0.00 | 0.00 | 0.50 |
| random | 5.23 | 0.008 | 0.00 | 0.00 | 0.50 |
| dlg_adapted | 7.71 | 0.047 | 0.50 | 0.667 | 0.21 |
| ig_adapted | 10.26 | 0.094 | 0.50 | 0.667 | 0.17 |

The PSNR ordering reverses between the two. At 60 iterations the control beats
every attack, which reads as "image fidelity comes from the smoothness prior";
at 2000 it does not, and `ig_adapted` recovers the private answer exactly in two
of four runs. **The first table measures the budget, not the attack.** Do not
quote a PSNR comparison from a run that has not been checked for budget
sensitivity; the repository's own real-model config uses 1000 iterations.

One effect survives the larger budget and is not an artifact: on the *question*
the control scores higher than either attack (0.50 against 0.17 and 0.21) and no
method matches it exactly. Optimization concentrates on the target, which is
where the loss is taken, and does worse than a plain prior on the question. This
is a fixture result about the pipeline, budgets and metrics, not a privacy claim
about any real model.

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
