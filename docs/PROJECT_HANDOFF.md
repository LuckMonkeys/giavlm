# Project Handoff

Updated: 2026-10-02 (Asia/Shanghai) · branch `main`; former
`gradient-diagnostics` work merged through `cf9411f`

Resume: read `AGENTS.md` first, then this file. Verify it is current:

```bash
git status --short && git log -8 --oneline
ps -eo pid,etime,cmd | rg 'examples.run_attack|utils.run_cmds'; nvidia-smi
```

## Status

- **Objective:** benchmark privacy leakage from federated VLM fine-tuning across
  architectures, datasets, federated algorithms, FedVLMBench strategies, adversary
  knowledge, and visual/text/semantic/cost metrics. First validate one full path:
  SLAKE + LLaVA-1.5-7B.
- **Stage:** method/protocol validation and diagnosis. The end-to-end pipeline runs;
  DLG and IG were exercised on SLAKE/LLaVA.
- **New implementation:** `dager_adapted` adds text-only discrete reconstruction
  under public images/lengths, with optional known questions. Original `dager`
  remains unimplemented. One authorized real-model `image_known` diagnostic and
  its post-commit token-ranking controls are now complete; they show token-set
  signal but failed sequence reconstruction. See the evidence below and
  `docs/validation.md` for controls and boundaries.
- **Validation (2026-10-02):** DAGER mathematical/interface tests, real GPU-4
  diagnostics, paired raw/public-residual top-50 scoring and post-commit random,
  wrong-text and corpus-frequency controls are complete. CPU regression: 247
  passed, 3 skipped; Ruff clean. Initial unrelated changes were preserved in
  commit `e1e9557`; local DAGER reference checkout is ignored like the other
  reference projects, with its own untracked PDF left intact.
- **Finding:** image reconstructions remain noise-like. With the ground-truth image
  and private-text lengths public, TAG recovered no question content in 18 runs;
  answer signal appeared only weakly under F-CL at round 0 (details below).
- **Diagnosis so far:** replay is exact. On 20 held-out images, matching loss has
  useful global ordering over the finite candidate bank, especially at trained
  round 10, but its local pixel-space descent direction is almost orthogonal to the
  direction back to the private image. This reconciles candidate discriminability
  with failed IG optimization. It does not establish unique recovery.

## Priorities

Current text-method direction: freeze the DAGER settings and repeat the token
ranking controls on multiple independent images before doing more sequence search.
The single-sample diagnostic contains real sample-associated token-set signal but
does not establish generalization. Initialization with zero LoRA-A gradients is
explicitly unsupported. The earlier TAG follow-up below remains a separate pending
comparison, not an active run.

1. **Interpret and extend the completed gradient-discriminability study.** The
   finite-bank result supports gradient loss as a global ranking signal, while the
   direction probes show why ordinary pixel-space descent fails. Before proposing a
   new optimizer, inspect the module/candidate-family results and test whether a
   coarse or derivative-free search can exploit the ordering without private
   reference selection. See the completed evidence and reports below.
2. IG random-init baseline under float32 candidates (`attack.image_dtype`, default
   since 2026-09-24). The parameter sweep used bf16 candidates; rerun before citing
   it. `attack.image_dtype=bfloat16` reproduces the old runs.
3. **Text-side leakage:** the completed `image_known` + known-length matrix found no
   question recovery and only weak F-CL/r0 answer signal. Next run a matched
   `image_question_known` + known-answer-length matrix (question fixed, answer
   private) across F-C/F-L/F-CL at rounds 0/10 to separate joint-search failure from
   an uninformative text objective. The older four-job schedule
   `run_yaml/slake_llava_image_question_known.yaml` remains unlaunched apart from a
   20-iteration pilot (`outputs/diagnostics/iqk_pilot`).
4. Before any paper claim from the strategy/round results: the IG grid on the new
   trained F-C snapshots (training done 2026-09-29, attack grid not run), 10–20 images instead of 3, more seeds, and more training rounds to see
   whether the F-L equilibrium keeps rising.
5. Expand only after an update carries recoverable image or text signal.
   - F-2stage; `private`, `question_known`, known-length conditions.
   - Enable LPIPS/CLIP and private-text metrics for formal runs.

## Evidence: Gradient Discriminability (2026-09-27, completed)

Private-reference diagnostic, not an attack benchmark. Configuration:
`configs/diagnostics/slake_llava_discriminability.yaml`; implementation:
`evaluation/gradient_discriminability.py`, `evaluation/discriminability_metrics.py`
and `evaluation/discriminability_report.py`. Conditions are F-C r0, F-L r0/r10 and
F-CL r0/r10. A 3-image pilot (native and fp32) preceded 20 `tune` development
images; the analysis was then frozen and applied to 20 independent `eval` images.
All score and direction jobs completed with return code 0 on physical GPUs 5/6.
Reports are under `outputs/diagnostics/discriminability/reports/`.

- **The scalar loss is globally discriminative on the finite candidate bank.** On
  validation, the probability that a near candidate has lower cosine loss than a
  far candidate is 0.860–0.877 at round 0 and 0.963–0.979 at round 10. Relative-L2
  gives 0.812–0.830 and 0.960–0.972 respectively. Image-level Spearman correlation
  between cosine loss and MSE is 0.71–0.74 at round 0 and 0.88–0.92 at round 10.
  No far candidate appears in the lowest-loss 1% in this constructed bank.
- **The relationship survives common-gradient centering.** Centered-cosine
  near-win probability is 0.878–0.898 at round 0 and 0.967–0.969 at round 10, so the
  ranking is not explained only by a shared image-independent component.
- **The local derivative does not point back to the truth.** Across six perturbation
  radii, the cosine between `-d(loss)/d(image)` and the exact truth displacement is
  approximately zero (about -3.8e-4 to 1.1e-3; sign direction about -4.5e-3 to
  4.5e-3). Depending on condition/objective/direction, only about 1–12% of tested
  radius/step combinations, averaged over five images, simultaneously reduce loss
  and MSE.
- **Conclusion:** gradient matching loss contains a useful global ordering signal,
  which becomes stronger after LoRA training, but local first-order optimization
  cannot readily exploit it in raw pixel space. This explains how noise gradients
  can look globally similar while reconstruction still fails. The result is limited
  to the declared finite candidates and does not prove identifiability or attack
  success.

## Evidence: Private-Reference Gradient Diagnostics (2026-09-23)

Outside the threat model (reads `capture/private`). Tools:
`python -m core.commands diagnose-gradient` (`evaluation/gradient_diagnostics.py`)
and attacks with `attack.init_source=private_reference`
(`docs/protocol.md#Attack Initialization`). Outputs: `outputs/diagnostics/step0/`,
`outputs/diagnostics/step1_pilot/` (the pilot ran under an earlier command that
has since been folded into `attack.init_*`; its labels say `oracle`). Condition: the 3 `iterations_2000` captures
(F-C, text_known, FedSGD, batch 1, **bf16** via `configs/slake_llava_ig_private.yaml`).

- **Replay is exact:** reloading data gives an identical image and public token
  IDs, and truth replay gives cosine loss ≤ 1.2e-7. The replay-mismatch hypothesis
  is refuted.
- **The objective prefers non-truth images.** IG reconstructions reach cosine loss
  0.09–0.18 at 2000 iterations and 0.027 in the 100k run (killed at 15150), with
  PSNR 6–9. Near-truth images score worse: +2% Gaussian noise (PSNR ~34) gives
  0.18–0.90, and a 1 px blur gives 0.36–0.65.
- **Features are not recovered:** the reconstruction's CLIP connector-input cosine
  to truth is 0.29–0.34, the same as uniform noise (0.33–0.34).
- **CLIP is extremely pixel-sensitive:** 0.2% pixel noise drops token cosine to
  about 0.87, and the result is the same in fp32 and bf16 (so it is not numerics).
  3 high-norm artifact tokens (~200 vs median 22) move under tiny perturbations.
- **Truth is not an attractor (pilot, 300 iterations, run-00000):** starting from
  a 3% noise mix (PSNR 38.9), the loss falls 0.30→0.057 while PSNR falls to 29.8.
  The wrong-update control ends at a similar PSNR of 29.1. From a 25% mix, PSNR
  stays flat at 20.5→20.1.

## Evidence: Private-Reference Init Grid (2026-09-24, completed)

`run_yaml/slake_llava_ig_reference_init.yaml`, outputs in
`outputs/diagnostics/reference_init/`. Same 3 SLAKE images and condition as the
sweep (F-C, text_known, bf16 victim, IG lr 0.003, TV 0.1, 2000 iterations); start
= (1-α)·truth + α·uniform noise; fp32 candidates unless noted. PSNR, 3 images:

| Job | Start | After 2000 iterations | SSIM start→end |
|---|---|---|---|
| α=0.01 | 46–49 | 19.3–20.0 | 0.91–0.99 → 0.16–0.30 |
| α=0.03 | 36–39 | 19.0–20.0 | 0.68–0.93 → 0.17–0.30 |
| α=0.1 | 26–29 | 18.5–19.4 | 0.49–0.62 → 0.16–0.29 |
| α=0.25 | 18–21 | 16.1–18.3 | 0.21–0.31 → 0.13–0.24 |
| α=0.03, bf16 candidates | 36–39 | 19.3–22.2 | → 0.37–0.46 |
| α=0.03, prior_only (TV, plain Adam) | 36–39 | 15.9–20.3 | → 0.15–0.73 |

- All 12 IG runs move away from the truth from the first checkpoint while the
  matching loss falls (to 0.04–0.12). Runs from α ≤ 0.1 all end near 19–20 dB.
- bf16 candidates drift more slowly because small steps round away; they are not a
  better attack. The fp32 fix does not change the conclusion.
- The TV-only control also loses PSNR at this step budget but keeps structure
  (SSIM 0.68–0.73 on 2 of 3 images); IG destroys it (SSIM 0.17–0.30). So the
  gradient term actively pulls away from the truth, not just the prior.
- Still open: a wrong-update control with the same start and sign steps, and a
  much smaller LR near the truth, to separate "objective prefers other images"
  from "sign-step noise floor".
- Initial loss vs α (`outputs/diagnostics/loss_curves/`, same checkpoint built in
  both dtypes, each against its own truth gradient; 3 images × 5 noise directions):
  fp32 grows smoothly (log-log slope ≈ 1.5–2, i.e. close to α²) only for
  α ≤ 1e-3 (PSNR ≳ 65 dB), then saturates at 0.3–1 and is non-monotone. The bf16
  victim has a dead zone: for α ≤ 1e-4 the perturbation is rounded away and the loss
  is exactly the truth's; at α = 3e-4 it jumps to ~0.02. Every α in the grid
  (0.01–0.25) is in the saturated regime. IG ends at loss 0.03–0.06 for every α,
  about the loss of truth perturbed by α ≈ 3e-4–1e-3, while at ~19 dB PSNR.

## Evidence: F-C vs F-L vs F-CL Reference Init (2026-09-25, completed)

`run_yaml/slake_llava_ig_reference_init_lora.yaml` (round 0, LoRA r=8, alpha=16, bf16
victim; otherwise identical to the F-C grid). Figures:
`outputs/diagnostics/loss_curves/{loss_curves_f_l,loss_curves_f_cl,strategy_comparison}.png`.

- PSNR/SSIM trajectories are indistinguishable across F-C, F-L and F-CL at every α
  (e.g. α=0.03 final PSNR median 19.1 / 20.1 / 19.4; SSIM 0.15–0.32 for all).
- Initial loss vs α: all three share the bf16 dead zone (α ≤ 1e-4) and jump at 3e-4.
  F-L's loss is about half of F-C's at every α (α=0.1: 0.32 vs 0.75; pure noise:
  0.63 vs 1.09), i.e. the LoRA update is *less* image-sensitive: a large
  image-independent (text-driven) component. F-CL sits close to F-C; at round 0 the
  connector holds 48–78% of the F-CL update's squared norm (measured).
- IG reaches lower matching loss under F-L (0.007–0.015 vs 0.03–0.06 for F-C) with the
  same PSNR: a lower loss does not mean a better reconstruction.
- Conclusion (round 0, r=8, n=3): LoRA gradients do not add image guidance. Rank
  alone at round 0 is unlikely to help (B gradients have rank ≤ r and mix image and
  text tokens). `server_round > 0` (nonzero A gradients) was tested next, see below.
- Parameter surface (LLaVA-1.5-7B, verified on the meta device): F-C 20,979,712
  (projector); F-L 2,498,560·r (r=8: 19,988,480; LoRA on all 224 LLaMA Linear layers
  q/k/v/o/gate/up/down, no lm_head, matching FedVLMBench's `find_all_linear_names`);
  F-CL the sum. FedVLMBench's own configs use alpha 32, dropout 0.05 and a
  Llama-3.2-3B + CLIP ViT-B/32 model; we use alpha 16, dropout 0 (deterministic
  replay) and LLaVA-1.5-7B.

## Evidence: Trained Global States, Rounds 0/5/10 (2026-09-27, completed)

F-L and F-CL each trained once (1 client, FedAvg + AdamW lr 1e-4, effective batch 8,
10 local steps/round, 10 rounds; `outputs/federation/slake_llava_{f_l,f_cl}_fedavg`),
then the reference-init grid at the round-5/10 snapshots
(`run_yaml/slake_llava_ig_reference_init_trained{,_fcl}.yaml`, capture protocol
unchanged: FedSGD, batch 1, text_known; `server_round` comes from the snapshot).
Figure: `outputs/diagnostics/loss_curves/rounds_comparison.png`.

- Training: effective batch 8 is batch 2 × gradient accumulation 4, chosen up front
  because GPU 6 had ~31 GB free. `train` loads each round's whole batch onto the GPU
  and evaluates it in one forward, so a full local epoch (FedVLMBench) is infeasible
  there. Loss per round: F-L 4.36 → 1.32, F-CL 3.80 → 1.16. LoRA B becomes nonzero
  (‖B‖_F ≈ 3 at round 5, ≈ 4 at round 10; per-module ‖ΔW‖_F median ≈ 0.2–0.3), and
  all 224 LoRA-A gradients are nonzero in the round-5/10 captures. Reproduce with
  `outputs/federation/slake_llava_{f_l,f_cl}_fedavg.protocol.json` plus the `--set`
  overrides recorded in `outputs/federation/slake_llava_f_cl_pipeline.sh`.
- F-C trained 2026-09-29 with the same settings on GPU 5
  (`outputs/federation/slake_llava_f_c_fedavg`, snapshots 0/5/10;
  `slake_llava_f_c_pipeline.sh`, training only). Loss per round 4.51 → 1.58
  (4.51, 3.70, 3.77, 3.61, 2.24, 2.10, 2.06, 3.11, 2.18, 1.58); the round-8 spike
  appears in F-L/F-CL too. The process reserved ~50 GB (nvidia-smi, shared GPU).
  No IG grid on these snapshots yet.

- F-L improves monotonically with training: final PSNR round 0 → 5 → 10 rises at every
  α and on all 3 images (α=0.01 median 19.9 → 21.6 → 24.1; SSIM ~0.2 → ~0.38). At
  round 10 the final PSNR is ~23.5–24 dB for α = 0.01–0.1 regardless of the start and
  the curve plateaus after ~1000 iterations while the TV-only control keeps falling;
  at α=0.25 two of three images end above their start (20.5→20.8, 20.8→21.3). This is
  the first sign of a weak restoring force, but the equilibrium (~24 dB) is still far
  from the truth and the first 50 iterations drop exactly as at round 0.
- F-CL does not: round 5 is ~1–3 dB above round 0, round 10 falls back to round-0
  levels at every α. Not explained by gradient-norm shares (round-10 F-CL: connector
  6–11%, LoRA-A 6–13%, LoRA-B 79–83% of ||g||²; F-L LoRA-A only 1–5%). Likely the
  trained connector changes the model state; n=3 cannot rule out chance.
- Initial loss vs α drops with training for both (F-L α=0.1: 0.32 / 0.039 / 0.013 at
  rounds 0/5/10; pure noise 0.63 / 0.59 / 0.31): the update becomes less image-
  sensitive and more dominated by an image-independent component. The bf16 dead zone
  (α ≤ 1e-4) and the jump at 3e-4 remain in every setting.

## Evidence: Text-Only Reconstruction with Public Images (2026-09-29, completed)

`run_yaml/slake_llava_image_known_lengths_states.yaml`; outputs under
`outputs/slake_llava_text_leakage/image_known_lengths_states/`. All six scheduler
jobs returned 0. Each condition uses the same 3 SLAKE `tune` images, one attack seed,
2000 iterations and one restart. The ground-truth image and per-sample
question/answer token lengths are declared public; question and answer contents are
private. The image is a fixed `Observation` input and is never optimized or scored.

| Condition | Question EM | Question ROUGE-L | Answer EM | Answer ROUGE-L | Answer word recall |
|---|---:|---:|---:|---:|---:|
| F-C r0 | 0 | 0 | 0 | 0 | 0 |
| F-C r10 | 0 | 0 | 0 | 0 | 0 |
| F-L r0 | 0 | 0 | 0 | 0 | 0 |
| F-L r10 | 0 | 0 | 0 | 0 | 0 |
| F-CL r0 | 0 | 0 | 0.333 | 0.467 | 0.500 |
| F-CL r10 | 0 | 0 | 0 | 0 | 0 |

- Question EM, ROUGE-1/L and word recall are zero in all 18 runs. All answer
  metrics are also zero outside F-CL/r0.
- Under F-CL/r0, one of three answers is recovered exactly and one partially
  (ROUGE-L 0.4, word recall 0.5). The exact recovery is the single-token target;
  the partial case has four target tokens. All random-start metrics are zero, but
  this sample size is too small to distinguish leakage from short-answer chance.
- Training to round 10 does not improve recovery: F-C and F-L remain at zero, and
  the weak F-CL/r0 answer signal disappears at r10.
- Lower matching loss does not imply recovery. For example, F-L/r10 run 0 reduces
  its raw TAG objective 1482.93→1174.37 (iteration 50→2000), while the best
  observable discrete score occurs at iteration 1250 (1.065 versus 1.413 at the
  end); both reconstructed fields still have zero overlap with the references.
  Raw TAG objectives are unnormalized and must not be compared across conditions.
- This is n=3 development evidence, not a paper result: the three runs are three
  images, not independent attack seeds. A paired known-question experiment and
  larger image/seed counts remain necessary.

## Evidence: First Real DAGER Diagnostic (2026-10-02, completed)

One authorized development sample was run on physical GPU 4 under trained F-L
round 10, FedSGD, LoRA rank 8, `image_question_known` and public target length.
Artifacts are under `outputs/diagnostics/dager_real_gpu4_f_l_r10/run-00000/`.
This is a functionality/signal diagnostic (`n=1`), not a reconstruction result for
the paper and not a threshold-tuning set.

- `public_residual`, joint Q/K/V and threshold 0.05 produced first- and
  second-layer quotient-space ranks 24 (public ranks 597 and 653 in hidden size
  4096). Neither space was saturated.
- It scanned all 32,064 vocabulary IDs. The filter returned 2 informative and 19
  public-overlap ambiguous candidates. The 2 informative candidates were both
  true private-answer token IDs (precision 1.0), but they covered only half of the
  four unique private content-token IDs (recall 0.5). Including ambiguous IDs gave
  precision 0.095, recall 0.5. This is evidence of partial token-set leakage only.
- Beam search filled four private content slots using 1,029 prefix evaluations in
  65 batches; attack work after model loading took 5.5 seconds and used no full
  gradient reranking. The submitted text had EM/ROUGE/word recall 0 and WER 1.0.
  Thus the first complete real reconstruction failed despite precise partial token
  detection.
- The first launch captured successfully but exposed a fresh-run/resume interface
  bug: `ExperimentRunner` passes `resume=true` to a new attack. Commit `0a61e94`
  makes DAGER restore only when its own checkpoint exists and adds a regression
  test. The attack was then run directly against the unchanged hash-checked public
  observation, followed by evaluation after result commitment. No retry changed
  the data, precision, training condition or search settings.
- GPU 4 was empty before the run, peaked during LLaVA capture/restore, and returned
  to 0 MiB afterward. There was no OOM. The stale `failed.json` in the attack
  directory records the pre-search interface failure; `result.json` and
  `evaluation.json` are the later committed successful execution artifacts.

The first diagnostic above used `image_question_known` and therefore treats the
question content as public. It is an answer-only upper bound, not the user's
intended unknown-question condition. A corrected run was subsequently completed
under `image_known + token_lengths_known` using the same sample and F-L round-10
state. Artifacts are under
`outputs/diagnostics/dager_real_gpu4_f_l_r10_image_known/run-00000/`.

- The public observation contains the image and lengths only (question 10 content
  tokens, target 4); public question/target text and token-ID fields are empty.
- The filter produced 1 informative and 9 ambiguous candidates. The informative
  candidate was a true private token (precision 1.0), but union recall was 1/14
  (0.071); question recall was 0 and target recall was 0.25.
- Beam search filled all 14 private content positions using 2,030 prefix checks in
  128 batches and 6.1 seconds after model load. Reconstructed question and target
  both had EM/ROUGE/word recall 0. Thus the intended real reconstruction failed,
  with substantially weaker token coverage than the known-question upper bound.
- GPU 4 returned to 0 MiB after completion. This remains `n=1` development
  evidence and no post-reference threshold adjustment was run.

### Paired public-direction ablation (`n=1`, 2026-10-02)

The corrected `image_known` public observation was reused without recapture to
compare the first-layer token ranking under `raw` and `public_residual`. Both
modes used joint Q/K/V, the same numerical rank settings, and an exact top 50 over
valid tokenizer IDs. Each attack committed its complete vocabulary scores before
evaluation read private token IDs; sequence work was capped after one prefix and
is not part of this comparison. Artifacts and the paired report are under
`outputs/diagnostics/dager_projection_ablation_valid_vocab/`.

- Raw top-50: precision 0.14, recall 0.50 (7/14 unique private IDs), question
  recall 0.40 and target recall 0.75. True-token ranks ranged from 2 to 26,249
  with median 389.5.
- Public-residual top-50: precision 0.26, recall 0.929 (13/14), question recall
  0.90 and target recall 1.0. It contained 9 public-overlap ambiguous IDs; true
  ranks ranged from 10 to 98 with median 16.5. The one miss ranked 98.
- The two top-50 sets intersected in 22 IDs (Jaccard 0.282), so public projection
  substantially reordered the vocabulary. On this sample it had a clear positive
  effect on token-set detection. This is not yet evidence for sequence recovery or
  generalization; repeat on a frozen multi-image development/validation cohort.
- An initial raw comparison was invalid: 49/50 entries were padded model embedding
  rows above the tokenizer vocabulary. LLaVA has 32,064 embedding rows but only
  32,002 tokenizer IDs. Commit `03f214f` excludes those rows, adds a regression
  test and a post-commit equal-top-k evaluator. The invalid artifacts under
  `outputs/diagnostics/dager_projection_ablation/` are retained only as an audit
  trail and must not be cited.

### Token-filter null controls (`n=1`, 2026-10-02)

The already committed full-vocabulary score artifacts were evaluated without new
gradient capture or attack tuning. The aggregate report is
`outputs/diagnostics/dager_projection_ablation_valid_vocab/filter_validation.json`.
It uses 100,000 random top-50 draws and 632 wrong SLAKE `tune` texts from other
image groups. No individual wrong-reference ID or text is retained in the report.

- Uniform random top-50 sampling expects 0.0219 of the 14 private IDs. The exact
  probability of at least the observed 7 raw hits is 4.99e-17; for the 13
  public-residual hits it is 8.41e-37. Neither of 100,000 draws reached either
  observation. The filter is therefore decisively not equivalent to uniform
  vocabulary sampling on this sample.
- The K curve also reflects ranking, not a lucky cutoff: public residual recovers
  9/14 IDs at K=20, 13/14 at K=50 and 14/14 at K=100 (average precision 0.336).
  Raw recovers 5/14, 7/14 and 7/14 respectively (average precision 0.187), and is
  still only 8/14 at K=1000.
- A corpus-frequency top-50 recovers all 10 question IDs but none of the four
  target IDs. Thus most question-token hits are vulnerable to a repeated-template
  explanation; uniform random is an insufficient baseline. Public residual still
  recovers all four target IDs, which are absent from that frequency baseline.
- After excluding wrong texts whose field token set exactly equals the current
  field, public residual's question recall is 0.90 versus a wrong-text median of
  about 0.46 (plus-one empirical tail 0.00168 over 593 texts); target recall is
  1.0 versus median 0 (tail 0.0306 over 619). Raw question recall 0.40 is not
  exceptional (tail 0.180), although its target recall 0.75 is (tail 0.0226).
- Only six length-matched wrong texts per field remain after removing exact token-
  set duplicates, so that control has a minimum attainable plus-one tail of 1/7
  and cannot establish significance. Descriptively, their target overlap is zero
  for both filters while the current target has 3/4 raw and 4/4 residual coverage.

Conclusion: the current DAGER scores contain genuine sample-associated token-set
signal, especially for the answer, and public-direction removal strengthens it.
They also strongly favor common/template tokens, so the 13/14 headline must not be
treated as 13 sample-specific discoveries. This remains post-hoc `n=1` development
evidence, not sequence recovery or generalization. The next valid step is a frozen
multi-image study with per-image random, wrong-text/frequency and random-subspace
controls.

### Surrogate visual-subspace sensitivity (`n=1`, GPU 5, 2026-10-02)

`evaluation/dager_surrogate_diagnostic.py` reuses the committed gradient above and
tests only the first-layer full-vocabulary ranking. Its GPU scoring phase never
reads private text; evaluation joins exact token IDs only after all 27 score
artifacts commit. Conditions are Raw, template-only, the true public image, three
matched-rank random subspaces, and seven true/random-pixel blend levels with three
fixed noise directions each. Artifacts are under
`outputs/diagnostics/dager_surrogate_gpu5_n1/`. The blends are an oracle
misspecification diagnostic, not an image-private attack.

- The template rank is 11, the incremental visual rank is 576 and every full
  visual condition has public rank 587. Matched random controls therefore remove
  exactly the same number of dimensions. Raw recovers 7/14 IDs at K=50 (AP 0.187),
  template-only 8/14 (AP 0.183), matched random spaces 8/14 (AP about 0.255), and
  the true image 13/14 (AP 0.336). No true token is marked ambiguous in any of
  these conditions.
- A 1% blend with random pixels is still 48.5 dB from the true image, but its
  aligned visual-feature cosine is only about 0.914 and its visual-subspace overlap
  about 0.690. It recovers 11/14 at K=50 (question 8/10, target 3/4; mean AP 0.287),
  already below the true image's 13/14 (9/10 and 4/4). A 3% blend is also 11/14;
  10% gives 10--11/14 (mean 10.67).
- From 25% through 100% random pixels, all nine conditions stabilize at 10/14
  (question 7/10, target 3/4), with mean AP 0.262 down to about 0.256. Even the
  pure-random images outperform Raw by three top-50 hits and the matched random
  Euclidean subspaces by two. Thus a generic subspace produced by the actual
  vision encoder removes useful visual-modality nuisance, while exact image
  content supplies the remaining three hits, including the fourth answer token.
- Across the 21 blends, visual-subspace overlap has descriptive Spearman
  correlation 0.826 with Recall@50 and 0.917 with AP. These are repeated measures
  of one gradient, so their nominal p-values are not inferential evidence. The
  pure-random image retains overlap about 0.544 and aligned feature cosine about
  0.727, confirming that the encoder maps unrelated pixels into a substantial
  shared visual manifold.
- Candidate-set size still matters: a representative pure-random image recovers
  9/14, 10/14, 11/14, 13/14 and 14/14 at K=20/50/100/200/500, while the true image
  reaches 14/14 by K=100. Raw remains at only 8/14 even at K=1000.

Conclusion: Raw is an operational no-image baseline, not a mathematical lower
bound. Wrong residual spaces can in principle harm ranking. On this sample,
however, any tested vision-encoder-derived surrogate helps over Raw, and the
benefit separates into a generic visual-manifold component plus an exact-image
component. The striking drop from true image to a 1% blend also shows that the
oracle advantage is fragile. Repeat the frozen design across independent images
and add natural public-image surrogates before proposing an image-unknown method.
GPU 5 was shared with an unrelated roughly 40.5 GiB allocation; this job used
about 16.3 GiB, completed without OOM, and released its allocation.

## Hypotheses (unverified)

- IG drift from near-truth starts comes from the objective preferring other images,
  not only from the sign-step noise floor (±lr per pixel per step). Untested control:
  same start and sign steps against a wrong update, and a much smaller LR.
- The F-L restoring effect grows with more training rounds (only rounds 0/5/10 seen).
- F-CL's round-10 fallback reflects the trained connector's model state rather than
  chance (n=3; the trained F-C snapshots, now available, would help separate this).
- The update's image-independent component (pure-noise gradients keep cosine
  0.4–0.7 with the truth's, higher after training) is what limits discriminability.

## Evidence: IG Parameter Sweep (completed)

Condition: LLaVA-1.5-7B, SLAKE `tune`, FedSGD, batch 1, F-C connector gradients,
`ig_adapted` with 1 restart, `text_known`, `attack.text_method=none`. 12 settings
× 3 images = 36 runs. Metrics: MSE/PSNR/SSIM (LPIPS/CLIP off).

| Sweep (others at default) | Values | PSNR | SSIM |
|---|---|---|---|
| Attack LR | 0.001 / 0.003 / 0.01 / 0.03 / 0.1 | 7.74 / 7.73 / 7.57 / 7.09 / 6.52 | ≈0.005–0.007 |
| TV weight | 0 / 0.001 / 0.01 / 0.1 | 7.73 / 7.73 / 7.73 / 7.78 | ≈0.007 |
| Iterations | 500 / 1000 / 2000 | 7.77 / 7.78 / 7.80 | 0.0071–0.0074 |

Conclusion: every setting is at noise level (SSIM ≈ 0.007), so parameter tuning
is not the bottleneck. These runs used bf16 candidate images (before
`attack.image_dtype`), so LR 0.001 was largely rounded away. The working point is LR 0.003, TV 0.1, 2000 iterations.
It is diagnostic only; n=3 is too small for paper claims.

Full numbers: `outputs/slake_llava_ig_parameter_sweep/{lr,tv,iterations,sweep}_report.json`,
`docs/ppt/ig_parameter_sweep_summary.xlsx`.

## Active / Pending Jobs

- `outputs/diagnostics/discriminability`: completed through pilot, development,
  frozen analysis, validation and final reports. Every scheduler job returned 0.
- At the user's request, `/home/zx/nas/gpu/train_stealth.py` is running on physical
  GPU 5 after study completion (`xlarge`, batch 70, fp32, 40% target utilization):
  it reports a 70,000 MiB allocation; PID and log are in
  `outputs/train_stealth_gpu5_70g.{pid,log}`.
- `run_yaml/slake_llava_ig_iteration_occ.yaml` (100000-iteration diagnostic): it
  died at iteration 15150; the scheduler's `running` row is stale.
- This discriminability study was restricted to physical GPUs 5 and 6, at most two
  GPUs and one worker per card. Check current assignments before any new launch.
- Memory: IG attack peaks ~29–32 GB (F-C/F-L/F-CL, bf16 victim, second-order);
  `train` with batch 2 ~20 GB. The trained-snapshot grids use `min_free_mib: 36000`;
  older schedules use 50000.
- Long runs are launched detached (`setsid nohup … & disown`) and watched with
  `outputs/diagnostics/analysis_scripts/watch_schedule.sh` / `watch_training.sh`,
  which exit on job completion, failure, a traceback/OOM, or a dead scheduler.

## Notes Git Cannot Express

- `Observation` is now schema v5 (declared public images). Captures under
  `outputs/slake_llava_ig_parameter_sweep/` and `outputs/diagnostics/step0`,
  `step1_pilot` are v4 and are rejected by `attack`/`diagnose-gradient`; recapture
  (deterministic, minutes) to reuse them. `outputs/diagnostics/reference_init*` and
  the trained-snapshot grids are v5. All reports and numbers above remain valid.
- `main` now includes the former `gradient-diagnostics` branch: diagnostics
  (`evaluation/gradient_diagnostics.py`, `diagnose-gradient`), attack starts
  (`attacks/init.py`, `attack.init_*`, `attack.init_text_*`), per-field knowledge
  with `image_known` / `image_question_known`, `attack.image_dtype`, Observation v5,
  `evaluation.trajectory` and `init_*` metrics, the staged discriminability study,
  their tests, these docs, and the associated `run_yaml/slake_llava_*` schedules. Tests:
  223 passed, 3 skipped; ruff clean (rechecked 2026-09-27).
- Analysis code that produced `outputs/diagnostics/loss_curves/*` is not in the repo:
  copies live in `outputs/diagnostics/analysis_scripts/` (loss curves, α vs victim
  dtype, CLIP feature sensitivity, parameter census, plots, per-job comparison,
  watchers). Fold them into `evaluation/` before relying on them for paper figures.
- `docs/ppt/02-preliminary-results.png` uses an illustrative reference image;
  replace it with `docs/ppt/slake-reference-*.png` in the final deck.

---
Update after major commits, finished experiment phases, or direction changes.
Refresh the header, link evidence rather than pasting logs or private data, keep
measured results separate from hypotheses, and delete superseded items.
