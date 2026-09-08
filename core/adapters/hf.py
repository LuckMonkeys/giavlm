"""Shared Hugging Face loading and fixed-block causal text protocol."""
import torch
from core.vlm_wrapper import VLMAdapter

class HFAdapter(VLMAdapter):
    def __init__(self, spec, training, backend=None, processor=None):
        super().__init__(spec, training)
        from transformers import (AutoProcessor, Blip2ForConditionalGeneration,
                                  LlavaForConditionalGeneration,
                                  Qwen2_5_VLForConditionalGeneration)
        classes = {"llava": LlavaForConditionalGeneration, "blip2": Blip2ForConditionalGeneration,
                   "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration}
        kwargs = {"revision": spec.revision, "local_files_only": spec.local_files_only}
        self.processor = processor or AutoProcessor.from_pretrained(spec.name, **kwargs)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # Use an existing non-EOS token as padding; no embedding resize or new weights.
            alternate = self.tokenizer.unk_token_id or self.tokenizer.bos_token_id
            if alternate is None or alternate == self.tokenizer.eos_token_id:
                alternate = 0
            self.tokenizer.pad_token_id = alternate
        placement = {}
        if spec.device_map:
            placement["device_map"] = spec.device_map
            if spec.max_memory:
                placement["max_memory"] = {int(k) if k.isdigit() else k: v for k, v in spec.max_memory.items()}
        self.backend = backend or classes[spec.family].from_pretrained(
            spec.name, torch_dtype=getattr(torch, spec.dtype), attn_implementation="eager",
            **kwargs, **placement)
        if spec.device_map and any(str(v) in {"cpu", "disk"} for v in self.backend.hf_device_map.values()):
            raise ValueError("CPU/disk weight offload is not supported for differentiable updates")
        config = self.backend.config
        if hasattr(config, "use_cache"):
            config.use_cache = False
        if spec.family == "blip2" and not config.use_decoder_only_language_model:
            raise ValueError("BLIP-2 adapter targets OPT causal LM checkpoints")
        image_processor = self.processor.image_processor
        self.register_buffer("image_mean", torch.tensor(image_processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_processor.image_std).view(1, 3, 1, 1))
        if spec.family in {"llava", "blip2"} and spec.image_size != config.vision_config.image_size:
            raise ValueError("Configured image_size must equal the vision encoder's native size")
        if spec.family == "qwen2_5_vl":
            factor = config.vision_config.patch_size * config.vision_config.spatial_merge_size
            if spec.image_size % factor:
                raise ValueError(f"Qwen image_size must be a multiple of {factor}")
        self.position_gradient_names = [name for name, _ in self.named_parameters()
                                        if "position_embedding" in name and not self.is_language(name)]

    def embedding(self):
        return self.backend.get_input_embeddings()

    def is_language(self, name):
        if self.spec.family == "qwen2_5_vl":
            return name.startswith(("backend.model.", "backend.lm_head."))
        return name.startswith("backend.language_model.")

    def public_embeddings(self, text, batch_size):
        ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=False),
                           device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return self.embedding().weight.new_zeros(batch_size, 0, self.embedding().weight.shape[1])
        return self.embedding()(ids).unsqueeze(0).expand(batch_size, -1, -1)

    def qwen_pixels(self, pixels):
        cfg = self.backend.config.vision_config
        p, merge, temporal = cfg.patch_size, cfg.spatial_merge_size, cfg.temporal_patch_size
        b, c, h, w = pixels.shape
        gh, gw = h // p, w // p
        pixels = pixels[:, None].expand(b, temporal, c, h, w)
        pixels = pixels.reshape(b, 1, temporal, c, gh // merge, merge, p, gw // merge, merge, p)
        pixels = pixels.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return pixels.reshape(b * gh * gw, c * temporal * p * p), torch.tensor(
            [[1, gh, gw]] * b, dtype=torch.long, device=self.device)

    def visual_embeddings(self, images):
        from core.adapters.llava import LlavaAdapter
        from core.adapters.blip2 import Blip2Adapter
        from core.adapters.qwen_vl import QwenVLAdapter
        classes = {"llava": LlavaAdapter, "blip2": Blip2Adapter, "qwen2_5_vl": QwenVLAdapter}
        return classes[self.spec.family].visual_embeddings(self, images)

    def target_logits(self, images, q, y):
        b = len(images)
        visual, grid = self.visual_embeddings(images)
        embedding = self.embedding().weight
        visual = visual.to(embedding.device)
        instruction = "Describe the image." if self.training_spec.task == "caption" else "Question: "
        prefix = "USER: " if self.spec.family == "llava" else ""
        before = self.public_embeddings(prefix, b)
        after = self.public_embeddings("\n" + instruction, b)
        pieces = [before, visual, after]
        if self.training_spec.task == "vqa":
            pieces.append(q.to(embedding.device) @ embedding)
        pieces.extend([self.public_embeddings("\nAnswer: ", b), y[:, :-1].to(embedding.device) @ embedding])
        embeds = torch.cat(pieces, 1)
        if self.spec.family == "qwen2_5_vl":
            # Use structural IDs only to calculate mRoPE; text content cannot affect positions.
            start = self.public_embeddings("<|vision_start|>", b)
            end = self.public_embeddings("<|vision_end|>", b)
            embeds = torch.cat([start, visual, end, *pieces[2:]], 1)
            cfg = self.backend.config
            structure = torch.zeros(embeds.shape[:2], dtype=torch.long, device=self.device)
            structure[:, 0] = cfg.vision_start_token_id
            structure[:, 1:1 + visual.shape[1]] = cfg.image_token_id
            structure[:, 1 + visual.shape[1]] = cfg.vision_end_token_id
            positions, _ = self.backend.get_rope_index(structure, image_grid_thw=grid)
            hidden = self.backend.model(inputs_embeds=embeds, position_ids=positions,
                                        use_cache=False, return_dict=True).last_hidden_state
            logits = self.backend.lm_head(hidden[:, -y.shape[1]:])
        else:
            logits = self.backend.language_model(inputs_embeds=embeds, use_cache=False,
                                                 return_dict=True).logits[:, -y.shape[1]:]
        return logits
