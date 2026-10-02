# Baseline Fidelity and Applicability

Every implementation in this package is an adaptation to the native-sft-v2 VLM
protocol. Do not label an adapted score as a numerical reproduction of an
original paper. Tune hyperparameters only on the designated tuning image groups.

| Name | Implemented mechanism | Explicit changes/conditions |
|---|---|---|
| dager_adapted | First-layer LoRA-A token span checks, second-layer prefix beam search, optional final gradient reranking | Single-sample LLaVA VQA, public image and token lengths, F-L/F-CL FedSGD; raw or public-input quotient subspaces; no exact-recovery guarantee |
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

## DAGER adaptation

`dager` remains the unimplemented original-paper entry. `dager_adapted` is an
independent implementation of the span-check mechanism in Petrov et al.,
*DAGER: Exact Gradient Inversion for Large Language Models*, NeurIPS 2024
(https://arxiv.org/abs/2405.15586). The local `dager-gradient-inversion/` checkout
is an ignored, read-only reference; no upstream model loading, pickle artifacts,
logging services or parameter-list indices are imported.

The first release requires `image_question_known` or `image_known`, declared
token lengths, one sample, single-device LLaVA/LLaMA with at least two decoder
blocks, F-L/F-CL and undefended FedSGD uploads. Required first/second-layer
LoRA-A tensors are resolved by name. Zero A gradients (including ordinary
LoRA initialization), missing uploads and incompatible protocols are explicit
`not_applicable` results. There is no fallback to LoRA-B, full-weight gradients,
TAG, a different dtype or a changed training strategy.

Select it with `attack=dager_adapted`; the preset sets `text_method=none` because
the method owns text recovery. Controls live under `attack.dager`:

- `mode=raw` tests normalized input vectors against the uploaded gradient row
  space. `mode=public_residual` first removes known input directions from both
  gradients and candidate vectors. Layer 0 uses all known input rows; layer 1
  uses only the causal prefix before the first unknown token.
- `projections=qkv` normalizes each raw Q/K/V gradient by its Frobenius norm
  before projection and stacking; `q`, `k` or `v` select one module.
- `analysis_dtype`, `rank_rtol`, `rank_atol` control decomposition only. Victim
  computation stays in its declared dtype. Upcasting cannot repair rounded uploads.
- `token_selection=threshold` keeps distances at most `token_threshold`; `topk`
  keeps the best `max_candidates` informative IDs. The same cap also applies
  after threshold filtering, with truncation recorded. Tokens annihilated by
  public projection remain ambiguous candidates outside this cap, not detections.
- `beam_width` bounds retained prefixes; `prefix_batch_size` and
  `vocab_chunk_size` bound work batches. Prefix scores sum second-layer span
  distances at private content positions; ambiguous positions contribute zero.
- `rerank_candidates=0` selects by prefix score. A positive value replays that
  many leading complete candidates and selects by relative-L2 residual on exactly
  the uploaded parameters. This is an explicit adaptation, not the original algorithm.

To attack a compatible, already-captured public observation later, use the staged
entry (substitute the public capture and a new output directory):

```bash
python -m core.commands attack --observation PUBLIC_CAPTURE --output NEW_ATTACK_OUTPUT --set attack.method=dager_adapted --set attack.text_method=none --set attack.checkpoint_interval=1
```

The observation supplies the authoritative victim model, dtype, training strategy,
knowledge and lengths; setting those fields in the attack config cannot make a
private image public or change a captured rank. A fresh ordinary LoRA initializer
will be rejected because its A gradients are zero. All numerical/search options
are recorded under result provenance and covered by the attack protocol hash.

Rank deficiency alone does not guarantee that true input vectors lie in the
observed gradient space, especially at LoRA rank 8. Zero numerical rank returns
`no_signal`; an empty vocabulary filter returns `no_candidates`. Saturated
subspaces are recorded in provenance. Bounded search can miss the true sequence.
`completed` means a full candidate was committed, not that it equals the truth.
Only offline mathematical and random-small-model interface tests have been run;
real token detection and real reconstruction have not been evaluated.

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
