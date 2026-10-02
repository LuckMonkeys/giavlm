# Protocol Contract

## Threat Model

The server passively observes an individual client's uploaded update and the
current model. No crop location, private sample ID, source file path,
optimizer hidden state, or private image is given to the attack. By default
candidates start from uniform noise; attacker-held start images are a declared
condition (see Attack Initialization). Per-sample
question/answer lengths are hidden unless `token_lengths_known` is enabled.
Model architecture/weights, tokenizer, task, fixed slot lengths,
preprocessing, optimizer hyperparameters, batch size, accumulation count and
local optimizer step count are public.
The fixture seed initializes only synthetic data and is not part of an attack
initialization heuristic. Benchmark data are treated as private observations
even though the underlying research datasets are publicly downloadable.

Knowledge conditions:

| Condition | VQA | Caption |
|---|---|---|
| private | Image, question, target private | Image and target private |
| question_known | Question public; image/target private | Not applicable |
| text_known | Question/target public; image private | Target public; image private |
| image_known | Image public; question/target private | Image public; target private |
| image_question_known | Image/question public; target private | Not applicable |

Each condition is a set of public fields (`KNOWLEDGE_FIELDS` in `core/config.py`).
A public field is a fixed input of the attack, never optimized and never scored;
evaluation lists the scored fields in `scored_fields`. A public image travels in
the observation as `public_images.safetensors`, exactly the preprocessed view the
client trained on. `image_question_known` isolates answer leakage, the most
attack-favorable text condition and hence an upper bound on text recovery.

Public text is the decoded sequence actually used by the model, after truncation.
It is excluded from recovery scores. Model utility may use all task annotations.
`token_lengths_known` is an independent capability: for VQA it publishes the
post-truncation question and target content-token counts; for captioning it
publishes only the target count. Counts exclude EOS/PAD. Since response-only loss
is public and contiguous, the target count uniquely determines the private loss
mask (content plus EOS); no mask is serialized.

## Attack Initialization

The start image is attacker knowledge like the text conditions above: it can make
a reconstruction resemble the reference with no help from the gradient. It is
therefore a recorded condition, never an implementation detail.

| `attack.init_source` | Start images | Use |
|---|---|---|
| `random` (default) | Uniform noise | Benchmark |
| `public_image` | `attack.init_images`: attacker-held images (safetensors `images`, one file, or a directory), e.g. a public same-modality template | Benchmark, as a stronger threat model |
| `private_reference` | The victim's own private images; defaults to the capture's `private/images.safetensors` | Diagnostic only; never a benchmark conclusion |

Private text has the matching `attack.init_text_source`:

| `attack.init_text_source` | Start tokens | Use |
|---|---|---|
| `random` (default) | Near-uniform logits | Benchmark |
| `public_text` | `attack.init_question` / `attack.init_target` templates, each applied to its private field only | Benchmark, as a stronger threat model |
| `private_reference` | The capture's own raw question/target, re-encoded exactly as captured | Diagnostic only |

A start token becomes logits `init_text_scale · one_hot(id)` plus the usual small
noise. `attack.init_text_perturbation=replace` swaps each content token for a
random non-special token with probability `init_text_level`; EOS/PAD positions are
kept, so a start adds no length information beyond its source. An initialization
must target a private field: an image start with a public image, or a text start
with all text public, is rejected.

Private candidate images are optimized as a master copy in `attack.image_dtype`
(default `float32`) and cast to the victim dtype only in the forward pass. The
victim model, its dtype and the observed update are unchanged. With a bfloat16
master copy, Adam/sign steps below about 0.004 round away for pixels at or above
0.5; runs made before this setting existed used the victim dtype and are
reproduced only with `attack.image_dtype=bfloat16`.

`attack.init_perturbation` (`uniform_mix`, `gaussian`, `blur`) and
`attack.init_level` perturb the start deterministically per restart; `uniform_mix`
level 1 is pure noise. The result condition records `init_source`, perturbation,
level and the start-image hash, so reports never pool different starts. Every
attack writes its actual start to `init.safetensors`; evaluation reports it as
`init_*` metrics, so the gradient's gain is the reconstruction score minus the
start score. For `public_image`, evaluation sets `init_near_reference` when a start
is within MSE 1e-3 (about 30 dB PSNR) of the reference; such a run is not a
public-knowledge result. With `evaluation.trajectory=true`, evaluation also scores
every committed checkpoint after the attack. `init.json` holds the decoded text
start and yields `init_question_*` / `init_target_*` metrics.

## DAGER Text Search

`dager_adapted` consumes the same schema-v5 Observation and keeps public images
fixed. It returns `images=None`, restores public question IDs unchanged, and
searches only private content tokens. Public lengths determine EOS/PAD; neither
reference text nor a private loss mask enters search. Image/text initialization
overrides are rejected. The generic `init_source=random` label is a neutral
configuration default here: this deterministic discrete search has no random
candidate start and writes no `init.*` artifacts.

The attack shares the victim's exact fixed-slot LLaVA layout. Question padding
remains in the attention context, as in native-sft-v2. Only the first decoder
block is executed for prefix checks; future unknown slots are never replayed.
Public image features and subspace bases are computed from the declared model
state. Known template tokens after a private question are not treated as known
contextual features at layer 1.

Search is bounded by `attack.seconds` and `attack.max_evaluations`. One evaluation
is one private-prefix extension or one full gradient replay, charged before the
operation, independently of batching. These units differ from optimizer steps;
costs separately expose vocabulary IDs scored, prefix candidates, forward
batches and full replays. `iterations` and optimizer learning rates are unused.
`checkpoint_interval` counts prefix forward batches; vocabulary chunks and stage
transitions are always checkpointed. Checkpoints use independent schema v1,
JSON/safetensors generations and an atomic pointer, bound to source, attack
options, public inputs, model fingerprint and uploaded tensors. Resume preserves
consumed time/evaluations. OOM and other exceptions persist failed state and
re-raise without a retry or protocol change.

Artifacts `token_candidates.safetensors` and its JSON metadata contain vocabulary
scores, selected IDs and a public-overlap ambiguity mask. `text_tokens.safetensors`
contains the final reconstructed IDs. Token filtering alone is not a completed
reconstruction; `budget_exhausted`, `no_candidates`, and `no_signal` have no fake
text output. Only evaluation opens private references after `result.json` is
committed. New captures store exact reference IDs in private
`text_tokens.safetensors`; older captures without it explicitly report token-ID
metrics unavailable, with no silent reconstruction of references by re-tokenizing.
Candidate-set metrics include ambiguous IDs; informative-detection metrics exclude
them. They measure the union of private text tokens, not order or field assignment.

## Native SFT V2

`native-sft-v2` uses family-specific LLaVA, BLIP-2 and Qwen2.5-VL public prompt
fragments and each checkpoint's image processor. Private question and response
content still occupies fixed maximum slots. True lengths remain hidden unless
the explicit length capability is enabled. Response-only causal loss includes EOS and excludes the prompt,
image positions, padding and positions after EOS. Hard one-hot and soft-token
candidates share this path.

For soft candidates, a token probability distribution supplies both input
embeddings and shifted soft target labels. EOS survival is differentiable and
comes from the candidate. With unknown lengths, the final slot is a public EOS
bound and earlier EOS is optimized. With known lengths, content positions remain
trainable while EOS and subsequent PAD positions are fixed. Scoring/restart
selection always replays **discrete**
candidate tokens, rather than reporting a fractional-label gradient match as
successful recovery. Pixel variables are constrained to [0,1].

Qwen uses a fixed square grid, differentiable temporal duplication/patchification,
the pretrained visual module and mRoPE positions computed from public structural
tokens. BLIP-2 keeps the visual and Q-Former paths differentiable even when their
parameters are frozen. All model families use eager attention.

## FedVLMBench Fine-Tuning Strategies

Fine-tuning strategy is independent of the federated algorithm. The `tuning`
Hydra group selects the trainable and uploaded parameter surface:

| Strategy | Active parameters |
|---|---|
| `f_c` | Multimodal connector only |
| `f_l` | LoRA parameters in language-model linear layers only |
| `f_cl` | Connector and language-model LoRA parameters concurrently |
| `f_2stage` | Connector first, then language-model LoRA |

The vision encoder, Q-Former where present, and language-model base weights stay
frozen in all four strategies. LoRA excludes the connector and `lm_head`. For
`f_2stage`, `server_round` is public protocol state: rounds below
`two_stage_connector_rounds` use the connector phase and later rounds use the
LLM-LoRA phase. The round recorded in an Observation therefore determines the
exact trainable and uploaded parameter set an attacker must replay.

## Upload Semantics

- `gradient`: mean supervised loss over one minibatch, differentiated with
  respect to exactly the trainable parameters.
- `client_delta`: theta_after - theta_before for the public SGD or AdamW rule.
  `local_steps` counts optimizer steps, each averaging
  `gradient_accumulation_steps` microbatches. For single-step SGD, delta = -lr * grad.
- Multistep reconstruction does not know the true slot ordering. It optimizes
  an ordered slot sequence to reproduce local training; evaluation treats the
  recovered image/text tuples as a set. Repeated actual training examples remain
  repeated slots if sampled during federation.
- Missing gradients for structurally unused trainable parameters are explicit
  zeros. Frozen parameters are absent, never exported as hypothetical gradients.
- `training.algorithm` is the single source of update semantics. FedSGD computes
  and uploads one-step gradients, averages them, and applies `-lr`; FedAvg computes
  and uploads local deltas, averages them, and adds them to the global model. Both
  use local dataset sizes as weights and the same initial global state for selected
  clients. FedAvg optimizer moments reset for every client and round. LoRA aggregation stays in A/B
  parameter space rather than substituting the product BA.

## Artifact Boundary

Schema-v5 `Observation` contains model/training specifications, model fingerprint, named
update tensors, only explicitly public text, and public images only under an
image-known condition (hash-checked `public_images.safetensors`). Loading verifies an allowlist,
metadata digest, tensor-file hash and parameter names. Schema-v4 model checkpoints
store only the strategy-owned mutable overlay in safetensors: connector for F-C,
LoRA for F-L, and both for F-CL/F-2stage. Restoration reloads the pinned external
base checkpoint, validates the overlay hash, names, shapes and dtypes, then verifies
the complete model fingerprint. Attack callbacks do not receive references.
Evaluation verifies that its reference artifact has the same observation ID as the
reconstruction.

In an attack `result.json`, `questions` (when the question is private) and
`targets` are reconstructed text, and `images.safetensors` / `image-*.png` are
reconstructed images. None of them is ground truth.

Result artifacts include a method/config signature, model state, attack seed,
actual update/local-backward/prior evaluation counts, runtime, memory, and status.
Optimizer recovery is serialized as JSON plus safetensors (no pickle). Immutable
checkpoint generations are committed by an atomic pointer update. Work after
the last committed checkpoint may be recomputed after an abrupt process loss;
external job logs are the source of truth for total billed wall time.

## Experimental Boundaries

No secure aggregation, malicious model modification, persistent Adam state, clipping,
DP guarantee, QLoRA, stochastic augmentation, LR scheduling or unknown optimizer is modeled.
No differential privacy claim follows from adding arbitrary Gaussian noise.
No natural-image benchmark proves OCR/sensitive-field recovery. Adding that
claim requires an independently labeled text-rich dataset and an OCR track.
