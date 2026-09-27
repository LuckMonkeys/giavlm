# Project Handoff

Updated: 2026-09-27 (Asia/Shanghai) · branch `gradient-diagnostics` (3 commits on
top of `main` at `451eea2`, not merged)

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
- **Finding:** reconstructions are noise-like, for images and private Q/A tokens.
- **Diagnosis so far (text_known, FedSGD batch 1, bf16 victim, n=3 images):** not a
  replay bug (truth replay is exact). The matching objective does not identify the
  image: images far from the truth reach lower loss than barely perturbed truths,
  and IG drifts away even from a start visually identical to the truth. At round 0,
  F-C, F-L and F-CL behave identically. After 10 rounds of real FedAvg training,
  F-L shows a weak restoring effect (IG settles near ~24 dB instead of ~19 dB);
  F-CL does not. Nothing approaches reconstruction.

## Priorities

1. **Next direction (user, 2026-09-27): gradient discriminability.** Before any more
   optimizer work, test question (1) systematically: does the matching loss track
   distance to the truth at all? Question (2), how to minimize it, only matters if
   (1) holds. No plan is written yet; the previous planning attempt was not
   completed. Useful evidence already in hand: initial loss vs α
   (`outputs/diagnostics/loss_curves/`), the high cosine of pure-noise gradients,
   and IG reconstructions with low loss but ~19 dB PSNR.
2. IG random-init baseline under float32 candidates (`attack.image_dtype`, default
   since 2026-09-24). The parameter sweep used bf16 candidates; rerun before citing
   it. `attack.image_dtype=bfloat16` reproduces the old runs.
3. Text-side leakage: `run_yaml/slake_llava_image_question_known.yaml` (4 jobs × 3
   images: `image_question_known` random / lengths known, `image_known`, and a
   private-reference answer start with 50% token replacement; prepared, not
   launched; a 20-iteration pilot passed, `outputs/diagnostics/iqk_pilot`).
   Consider running it on the trained F-L snapshots, where the update is dominated
   by a text-driven component.
4. Before any paper claim from the strategy/round results: trained F-C snapshots
   (missing), 10–20 images instead of 3, more seeds, and more training rounds to see
   whether the F-L equilibrium keeps rising.
5. Expand only after an update carries recoverable image or text signal.
   - F-2stage; `private`, `question_known`, known-length conditions.
   - Enable LPIPS/CLIP and private-text metrics for formal runs.

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

## Hypotheses (unverified)

- IG drift from near-truth starts comes from the objective preferring other images,
  not only from the sign-step noise floor (±lr per pixel per step). Untested control:
  same start and sign steps against a wrong update, and a much smaller LR.
- The F-L restoring effect grows with more training rounds (only rounds 0/5/10 seen).
- F-CL's round-10 fallback reflects the trained connector's model state rather than
  chance (n=3; trained F-C snapshots would help separate this).
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

- None running (trained-snapshot grids for F-L and F-CL finished 2026-09-27 ~00:40).
- `run_yaml/slake_llava_ig_iteration_occ.yaml` (100000-iteration diagnostic): it
  died at iteration 15150; the scheduler's `running` row is stale.
- User GPU assignment for this line of work: GPU 6 (F-L / general), GPU 7 (F-CL).
  Other users share both; check free memory before launching.
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
- Branch `gradient-diagnostics` (not merged into `main`) holds: diagnostics
  (`evaluation/gradient_diagnostics.py`, `diagnose-gradient`), attack starts
  (`attacks/init.py`, `attack.init_*`, `attack.init_text_*`), per-field knowledge
  with `image_known` / `image_question_known`, `attack.image_dtype`, Observation v5,
  `evaluation.trajectory` and `init_*` metrics, their tests, these docs, and five
  `run_yaml/slake_llava_*` schedules. Tests: 205 pass; 2 CPU-venv failures need the
  missing `datasets` package; ruff clean (rechecked 2026-09-27).
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
