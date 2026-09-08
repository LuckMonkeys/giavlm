import math
import numpy as np
from skimage.metrics import structural_similarity

def image_metrics(reference, prediction):
    x, y = reference.float().cpu().numpy(), prediction.float().cpu().numpy()
    mse = float(np.mean((x - y) ** 2))
    size = min(7, x.shape[-1], x.shape[-2])
    size = size if size % 2 else size - 1
    if size < 3:
        raise ValueError("SSIM requires image dimensions of at least 3")
    return {"mse": mse, "psnr": -10 * math.log10(mse) if mse > 0 else None,
            "psnr_infinite": mse == 0,
            "ssim": float(structural_similarity(x, y, data_range=1, channel_axis=0, win_size=size))}
