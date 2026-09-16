# Protocol Contract

## Threat Model

The server passively observes an individual client's uploaded update and the
current model. No template, crop location, private sample ID, source file path,
question/answer length, optimizer hidden state, or ground-truth image is given
to the attack. Model architecture/weights, tokenizer, task, fixed slot lengths,
preprocessing, SGD learning rate, batch size and local step count are public.
The fixture seed initializes only synthetic data and is not part of an attack
initialization heuristic. Benchmark data are treated as private observations
even though the underlying research datasets are publicly downloadable.

Knowledge conditions:

| Condition | VQA | Caption |
|---|---|---|
| private | Image, question, target private | Image and target private |
| question_known | Question public; image/target private | Not applicable |
| text_known | Question/target public; image private | Target public; image private |

Public text is the decoded sequence actually used by the model, after truncation.
It is excluded from recovery scores. Model utility may use all task annotations.

## Fixed-Block Training

`fixed-block-eos-v1` is a controlled, deterministic fine-tuning format, not a
claim of native LLaVA/Qwen instruction-tuning reproduction. Image features,
fixed public prompt fragments, question slots (VQA only), an answer separator,
and autoregressively shifted target slots are concatenated. Language attention
is causal. All question slots occupy fixed positions; slots after EOS contain
padding. Target loss includes EOS and excludes padding/positions after EOS.

For soft candidates, a token probability distribution supplies both input
embeddings and shifted soft target labels. EOS survival is differentiable and
comes from the candidate. The final slot is a public EOS bound; earlier EOS is
unknown and optimized. Scoring/restart selection always replays **discrete**
candidate tokens, rather than reporting a fractional-label gradient match as
successful recovery. Pixel variables are constrained to [0,1].

Qwen uses a fixed square grid, differentiable temporal duplication/patchification,
the pretrained visual module and mRoPE positions computed from public structural
tokens. BLIP-2 keeps the visual and Q-Former paths differentiable even when their
parameters are frozen. All model families use eager attention.

## Upload Semantics

- `gradient`: mean supervised loss over one minibatch, differentiated with
  respect to exactly the trainable parameters.
- `client_delta`: theta_after - theta_before for the public SGD rule. There is
  one candidate minibatch per local step. For single-step SGD, delta = -lr * grad.
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
  clients. LoRA aggregation stays in A/B
  parameter space rather than substituting the product BA.

## Artifact Boundary

`Observation` contains model/training specifications, model fingerprint, named
update tensors and only explicitly public text. Loading verifies an allowlist,
metadata digest, tensor-file hash and parameter names. Model checkpoints use
safetensors and their own weight fingerprint. Attack callbacks do not receive
references. Evaluation verifies that its reference artifact has the same
observation ID as the reconstruction.

Result artifacts include a method/config signature, model state, attack seed,
actual update/local-backward/prior evaluation counts, runtime, memory, and status.
Optimizer recovery is serialized as JSON plus safetensors (no pickle). Immutable
checkpoint generations are committed by an atomic pointer update. Work after
the last committed checkpoint may be recomputed after an abrupt process loss;
external job logs are the source of truth for total billed wall time.

## Experimental Boundaries

No secure aggregation, malicious model modification, hidden Adam state, clipping,
DP guarantee, QLoRA, stochastic augmentation or unknown optimizer is modeled.
No differential privacy claim follows from adding arbitrary Gaussian noise.
No natural-image benchmark proves OCR/sensitive-field recovery. Adding that
claim requires an independently labeled text-rich dataset and an OCR track.
