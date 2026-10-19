"""Qwen3.5 native fixed-slot SFT and first-two-DeltaNet input replay.

Transformers' Qwen3.5 classes are imported only during model construction so
existing experiments can keep their original environment.
"""
import torch

from core.adapters.hf import HFAdapter
from core.adapters.llava import LlavaTextView
from core.vlm_wrapper import VLMAdapter


USER = "<|im_start|>user\n"
ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n"


def qwen35_input_pieces(adapter, visual, question, response):
    batch = len(visual)
    instruction = (adapter.public_embeddings("\nDescribe the image.", batch)
                   if adapter.training_spec.task == "caption"
                   else torch.cat([adapter.public_embeddings("\n", batch), question], 1))
    return [adapter.public_embeddings(USER, batch),
            adapter.public_embeddings("<|vision_start|>", batch), visual,
            adapter.public_embeddings("<|vision_end|>", batch), instruction,
            adapter.public_embeddings(ASSISTANT, batch), response]


class Qwen35Adapter(HFAdapter):
    def __init__(self, spec, training, backend=None, processor=None):
        VLMAdapter.__init__(self, spec, training)
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if spec.device_map:
            raise ValueError("Qwen3.5 adapter currently requires a single device")
        kwargs = {"revision": spec.revision, "local_files_only": spec.local_files_only}
        self.processor = processor or AutoProcessor.from_pretrained(spec.name, **kwargs)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None or self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            raise ValueError("Qwen3.5 requires distinct existing PAD and EOS tokens")
        self.backend = backend or Qwen3_5ForConditionalGeneration.from_pretrained(
            spec.name, dtype=getattr(torch, spec.dtype), attn_implementation="eager", **kwargs)
        self.backend.config.text_config.use_cache = False
        image_processor = self.processor.image_processor
        self.register_buffer("image_mean", torch.tensor(image_processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_processor.image_std).view(1, 3, 1, 1))
        config = self.backend.config.vision_config
        factor = config.patch_size * config.spatial_merge_size
        if spec.image_size % factor:
            raise ValueError(f"Qwen3.5 image_size must be a multiple of {factor}")
        self.position_gradient_names = []

    def is_language(self, name):
        return name.startswith(("backend.model.language_model.", "backend.lm_head."))

    def is_connector(self, name):
        return name.startswith("backend.model.visual.merger.")

    def prepare_image(self, image):
        return VLMAdapter.prepare_image(self, image)

    def visual_embeddings(self, images):
        pixels = ((images - self.image_mean) / self.image_std).to(self.dtype)
        pixels, grid = self.qwen_pixels(pixels)
        features = self.backend.model.visual(pixels, grid_thw=grid).pooler_output
        return features.reshape(len(images), -1, features.shape[-1]), grid

    def position_ids(self, pieces, grid):
        shape = (len(pieces[0]), sum(piece.shape[1] for piece in pieces))
        structure = torch.zeros(shape, device=pieces[0].device, dtype=torch.long)
        modalities = torch.zeros_like(structure)
        visual_start = pieces[0].shape[1] + pieces[1].shape[1]
        modalities[:, visual_start:visual_start + pieces[2].shape[1]] = 1
        positions, _ = self.backend.model.get_rope_index(
            structure, modalities, image_grid_thw=grid)
        return positions

    def target_logits(self, images, q, y):
        visual, grid = self.visual_embeddings(images)
        weights = self.embedding().weight
        pieces = qwen35_input_pieces(
            self, visual, q.to(weights.device) @ weights, y[:, :-1].to(weights.device) @ weights)
        hidden = self.backend.model.language_model(
            inputs_embeds=torch.cat(pieces, 1), position_ids=self.position_ids(pieces, grid),
            use_cache=False, return_dict=True).last_hidden_state
        return self.backend.lm_head(hidden[:, -y.shape[1]:])


class Qwen35TextView(LlavaTextView):
    """Reuse fixed-slot bookkeeping; override all architecture-specific replay.

    The first two DeltaNet blocks do not use RoPE. Their unmasked causal input
    replay therefore needs neither later blocks nor any unknown future slot.
    """

    @torch.no_grad()
    def __init__(self, adapter, observation):
        self.adapter = adapter
        self.decoder = adapter.backend.model.language_model
        if self.decoder.config.layer_types[:2] != ["linear_attention"] * 2:
            raise ValueError("Qwen3.5 text view requires two initial linear-attention blocks")
        self.device = adapter.embedding().weight.device
        model, training = observation.model, observation.training
        self.question_content_length = observation.public_question_lengths[0]
        self.question_is_public = training.question_public
        self.question_ids = self._slots(model.question_length, adapter.eos, adapter.pad,
                                        self.question_content_length)
        if training.question_public:
            self.question_ids = list(observation.public_question_ids[0])
            if adapter.content_lengths(torch.tensor([self.question_ids])) != observation.public_question_lengths:
                raise ValueError("Public question IDs differ from declared public lengths")
        self.target_ids = self._slots(model.target_length, adapter.eos, adapter.pad,
                                      observation.public_target_lengths[0])
        visual, _ = adapter.visual_embeddings(observation.public_images.to(
            device=self.device, dtype=adapter.dtype))
        question = adapter.embedding()(torch.tensor([self.question_ids], device=self.device))
        response = adapter.embedding()(torch.tensor([self.target_ids[:-1]], device=self.device))
        pieces = qwen35_input_pieces(adapter, visual, question, response)
        self.embeddings = torch.cat(pieces, 1)
        self.visual_start = pieces[0].shape[1] + pieces[1].shape[1]
        self.visual_stop = self.visual_start + visual.shape[1]
        question_start = sum(piece.shape[1] for piece in pieces[:4]) + adapter.public_embeddings("\n", 1).shape[1]
        response_start = sum(piece.shape[1] for piece in pieces[:-1])
        self.slots = []
        if training.question_private:
            self.slots.extend(("questions", i, question_start + i)
                              for i in range(self.question_content_length))
        self.slots.extend(("targets", i, response_start + i)
                          for i in range(observation.public_target_lengths[0]))
        known = torch.ones(self.embeddings.shape[1], dtype=torch.bool, device=self.device)
        for _, _, position in self.slots:
            known[position] = False
        self.known_mask = known
        self.public_layer0 = self.layer_input(self.embeddings[:, known], 0)[0]
        end = self.slots[0][2] if self.slots else self.embeddings.shape[1]
        self.public_layer1 = self.layer_input(self.embeddings[:, :end], 1)[0]

    @torch.no_grad()
    def layer_input(self, embeddings, layer):
        if layer == 0:
            return self.decoder.layers[0].input_layernorm(embeddings)
        if layer != 1:
            raise ValueError("Qwen3.5 text view exposes only the first two DeltaNet inputs")
        hidden = self.decoder.layers[0](
            embeddings, position_embeddings=None, attention_mask=None,
            past_key_values=None, use_cache=False)
        return self.decoder.layers[1].input_layernorm(hidden)

    def text_prior_ids(self, prefixes):
        depth = len(prefixes[0])
        if any(len(p) != depth for p in prefixes) or depth >= len(self.slots):
            raise ValueError("Prior requires equal-length unfinished parent prefixes")
        encode = self.adapter.tokenizer.encode
        before = encode(USER, add_special_tokens=False) + encode("\n", add_special_tokens=False)
        boundary = encode(ASSISTANT, add_special_tokens=False)
        rows = []
        for prefix in prefixes:
            if self.question_is_public:
                question = self.question_ids[:self.question_content_length]
                answer = prefix
            else:
                question = prefix[:self.question_content_length]
                answer = prefix[self.question_content_length:]
            values = before + question
            if self.slots[depth][0] == "targets":
                values += boundary + answer
            rows.append(values)
        return torch.tensor(rows, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def text_prior_nll(self, prefixes, candidates):
        ids = self.text_prior_ids(prefixes)
        hidden = self.decoder(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[:, -1]
        logits = self.adapter.backend.lm_head(hidden).float()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite text-prior logits")
        return -logits.log_softmax(dim=-1)[:, candidates], ids.numel()
