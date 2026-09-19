# Baseline Fidelity and Applicability

Every implementation in this package is an adaptation to the native-sft-v2 VLM
protocol. Do not label an adapted score as a numerical reproduction of an
original paper. Tune hyperparameters only on the designated tuning image groups.

| Name | Implemented mechanism | Explicit changes/conditions |
|---|---|---|
| dlg_adapted | Sum of squared parameter-gradient residuals; L-BFGS | Soft autoregressive targets; optimized EOS by default or fixed EOS under `token_lengths_known`; one inner L-BFGS step per outer iteration |
| ig_adapted | Global cosine residual, TV, Adam, signed image gradients | Alternating private text steps; selected common text component |
| april_adapted | Squared residual plus positional-gradient cosine | Optimization variant only; no closed-form or classification label rule; gradient uploads and learned visual positions required |
| gradvit_adapted | Layer L2 residual, external CNN BN statistics, patch-edge prior, TV, two-stage loss schedule | Explicit local prior checkpoint; no registration ensemble; classification labels replaced by text candidates |
| gi_dqa_adapted | Layer mean-square plus cosine residual, spatial/channel TV, Laplacian and Gaussian terms, early cosine-decayed priors | Entire RGB image from noise; no template or region mask; public/optimized text component instead of legacy extraction |
| tag_adapted | Soft token optimization using L2 + 0.01 L1 matching on text steps | Fixed public length bound; EOS optimized unless lengths are explicitly public; no true labels or embedding-row oracle |
| lamp_adapted | Continuous token candidates plus discrete swap/move proposals scored with update residual and public LM NLL | Soft vocabulary coordinates, two permutation proposals per field, autoregressive VLM targets; not original BERT code |

Visual optimization and textual optimization alternate for Adam-based variants.
DLG jointly updates all candidate variables with L-BFGS. With `text_known`, no
private text parameters are created and no text prior is loaded. All methods
receive the same observation; APRIL does not change the client's trainable set.
On LoRA-only uploads it is marked `not_applicable`, not assigned a zero score.

GradViT's schedule activates the external image prior in the second half and
halves gradient matching at that point. It has no implicit fallback to TV-only
optimization. Its original external prior was MoCo-v2 ResNet50; provide that
backbone or explicitly name a different-prior adaptation in experiment notes.
BN statistics are from the public prior network, never from victim activations.

LAMP's public LM uses its own tokenizer on decoded candidate text. The prior
scores only discrete proposals, so tokenizer mismatch is not approximated with
a fake differentiable vocabulary mapping. `tiny-public-bigram` is only an
offline unit-test fixture. It is rejected for real VLM models. For study quality,
include TAG/LAMP comparisons under a fixed visual method and all-private track.

Original GI-DQA template reconstruction remains in the reference project and is
not integrated into the natural-image score table. The legacy implementation has
different knowledge/initialization assumptions and needs an independent fidelity
audit before its numerical outputs can be compared with the new adapters.

## Audited Extensions

- DAGER requires an architecture/gradient-rank and token-subspace analysis before
  transfer to fused VLMs or LoRA; generic exact recovery is not claimed.
- MMGIA's modality-specific stage and shared latent-space refinement cannot be
  assumed identical to a fused autoregressive VLM objective.
- These two entries return `not_implemented`, preserving the distinction from
  architecture/protocol incompatibility. They are not part of the core release's
  implemented-method count.

## Sources

- DLG: https://github.com/mit-han-lab/dlg
- Inverting Gradients: https://github.com/JonasGeiping/invertinggradients
- APRIL: https://arxiv.org/abs/2112.14087
- GradViT: https://arxiv.org/abs/2203.11894
- GI-DQA: https://proceedings.mlr.press/v267/hemo25a.html
- TAG: https://aclanthology.org/2021.findings-emnlp.305/
- LAMP: https://github.com/eth-sri/lamp
- DAGER: https://openreview.net/pdf?id=CrADAX7h23
- MMGIA: https://www.ijcai.org/proceedings/2025/886

The adapters implement the described mathematical mechanisms independently;
the original codebases are not copied into this package.
