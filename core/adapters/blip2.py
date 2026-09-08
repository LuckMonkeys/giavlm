import torch
from core.adapters.hf import HFAdapter


class Blip2Adapter(HFAdapter):
    def visual_embeddings(self, images):
        pixels = ((images - self.image_mean) / self.image_std).to(self.dtype)
        visual = self.backend.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state
        queries = self.backend.query_tokens.expand(len(images), -1, -1)
        output = self.backend.qformer(
            query_embeds=queries, encoder_hidden_states=visual,
            encoder_attention_mask=torch.ones(visual.shape[:2], device=visual.device),
            return_dict=True).last_hidden_state
        return self.backend.language_projection(output), None
