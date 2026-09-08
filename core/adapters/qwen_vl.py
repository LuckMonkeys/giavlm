from core.adapters.hf import HFAdapter


class QwenVLAdapter(HFAdapter):
    """Qwen2.5-VL adapter; not a claim of original Qwen-VL compatibility."""

    def visual_embeddings(self, images):
        pixels = ((images - self.image_mean) / self.image_std).to(self.dtype)
        pixels, grid = self.qwen_pixels(pixels)
        features = self.backend.visual(pixels, grid_thw=grid)
        return features.reshape(len(images), -1, features.shape[-1]), grid
