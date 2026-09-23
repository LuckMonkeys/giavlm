# Project Handoff

Updated: 2026-09-23 (Asia/Shanghai) · HEAD `a376198` on `main`

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
- **Stage:** method/protocol validation. The end-to-end pipeline runs; DLG and IG
  were exercised on SLAKE/LLaVA.
- **Finding:** reconstructions are noise-like, for images and private Q/A tokens.
- **Open question:** is this a replay/protocol mismatch, or does the attack just
  not work on these VLM updates?

## Priorities

1. Validate protocol correctness.
   - Native LLaVA vs adapter logits/loss on identical hard inputs.
   - Captured client gradients vs attack-replay gradients.
   - Meaningful gradient reaches candidate images and private tokens.
2. Diagnose the attack path.
   - Log gradient norms, matching loss, candidate update magnitudes.
   - Controlled initialization; wrong-update / negative controls.
3. Expand only after correctness holds.
   - SLAKE tuning to ~20–50 samples, multiple seeds.
   - F-C, F-L, F-CL, then F-2stage.
   - `private`, `question_known`, `text_known`, known-length conditions.
   - Enable LPIPS/CLIP and private-text metrics for formal runs.

## Hypotheses (unverified)

- Replay differs from capture (prompt/mask/preprocessing/dtype), so matching
  optimizes the wrong target.
- The F-C connector gradient with batch 1 carries too little image signal for
  pixel-space IG through the frozen vision tower.

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
is not the bottleneck. The working point is LR 0.003, TV 0.1, 2000 iterations.
It is diagnostic only; n=3 is too small for paper claims.

Full numbers: `outputs/slake_llava_ig_parameter_sweep/{lr,tv,iterations,sweep}_report.json`,
`docs/ppt/ig_parameter_sweep_summary.xlsx`.

## Active / Pending Jobs

- `run_yaml/slake_llava_ig_iteration_occ.yaml` (100000-iteration diagnostic): the
  scheduler records `running`, but no process was alive on 2026-09-23. Treat it as
  stale.
- Scheduling uses `min_free_mib: 50000`, which is about what the LLaVA path needs.

## Notes Git Cannot Express

- The new `run_yaml/slake_llava_ig_*` sweeps, `docs/ppt/`, and `run.sh` edits are
  uncommitted user work. Full tests were not rerun after them.
- `docs/ppt/02-preliminary-results.png` uses an illustrative reference image;
  replace it with `docs/ppt/slake-reference-*.png` in the final deck.

---
Update after major commits, finished experiment phases, or direction changes.
Refresh the header, link evidence rather than pasting logs or private data, keep
measured results separate from hypotheses, and delete superseded items.
