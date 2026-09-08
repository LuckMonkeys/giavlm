import numpy as np
from PIL import Image
import torch

class OptionalMetrics:
    def __init__(self, spec, device="cpu"):
        self.spec, self.device = spec, device
        self.status = {"lpips": "disabled", "clip": "disabled"}
        self.lpips = self.clip = None
        if spec.lpips:
            import lpips
            # torchvision handles the pretrained backbone cache; doctor checks it before formal runs.
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
            self.status["lpips"] = "enabled:alex"
        if spec.clip:
            if not spec.clip_revision:
                raise ValueError("Pin evaluation.clip_revision before enabling CLIP metrics")
            from transformers import CLIPModel, CLIPProcessor
            kwargs = {"revision": spec.clip_revision, "local_files_only": True}
            self.clip = CLIPModel.from_pretrained(spec.clip_model, **kwargs).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(spec.clip_model, **kwargs)
            self.status["clip"] = f"enabled:{spec.clip_model}@{spec.clip_revision}"

    def score(self, ref_image, pred_image, ref_text, pred_text, task):
        result = {}
        with torch.no_grad():
            if self.lpips is not None:
                x, y = ref_image[None].to(self.device) * 2 - 1, pred_image[None].to(self.device) * 2 - 1
                if min(x.shape[-2:]) < 64:
                    raise ValueError("LPIPS needs >=64px inputs; do not silently resize the tiny fixture")
                result["lpips"] = float(self.lpips(x, y).item())
            if self.clip is not None:
                images = [Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
                          for x in [ref_image, pred_image]]
                inputs = self.processor(images=images, text=[ref_text, pred_text], padding=True,
                                        truncation=True, return_tensors="pt").to(self.device)
                output = self.clip(**inputs)
                im = torch.nn.functional.normalize(output.image_embeds, dim=-1)
                tx = torch.nn.functional.normalize(output.text_embeds, dim=-1)
                result["clip_image_similarity"] = float((im[0] * im[1]).sum())
                if task == "caption":
                    result.update({"clip_recon_image_true_text": max(0, float(im[1] @ tx[0])) * 2.5,
                                   "clip_true_image_recon_text": max(0, float(im[0] @ tx[1])) * 2.5,
                                   "clip_recon_image_recon_text": max(0, float(im[1] @ tx[1])) * 2.5})
        return result
