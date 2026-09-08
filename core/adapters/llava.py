from core.adapters.hf import HFAdapter


class LlavaAdapter(HFAdapter):
    def visual_embeddings(self, images):
        pixels = ((images - self.image_mean) / self.image_std).to(self.dtype)
        return self.backend.get_image_features(
            pixels, self.backend.config.vision_feature_layer,
            self.backend.config.vision_feature_select_strategy), None
