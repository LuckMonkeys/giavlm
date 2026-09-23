# Protocol Contract

## Threat Model

The server passively observes an individual client's uploaded update and the
current model. No template, crop location, private sample ID, source file path,
optimizer hidden state, or ground-truth image is given to the attack. Per-sample
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

Public text is the decoded sequence actually used by the model, after truncation.
It is excluded from recovery scores. Model utility may use all task annotations.
`token_lengths_known` is an independent capability: for VQA it publishes the
post-truncation question and target content-token counts; for captioning it
publishes only the target count. Counts exclude EOS/PAD. Since response-only loss
is public and contiguous, the target count uniquely determines the private loss
mask (content plus EOS); no mask is serialized.

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

Schema-v4 `Observation` contains model/training specifications, model fingerprint, named
update tensors and only explicitly public text. Loading verifies an allowlist,
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
