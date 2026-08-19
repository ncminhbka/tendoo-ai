"""
Attention-Guided Endogenous Text-Region Localization for FreeText.
arXiv:2601.00535 Section 3.1.
Extracts spatial attribution from DiT cross-attention, performs layer selection,
and applies topology-aware refinement (Otsu thresholding, connected components).
"""

import sys
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def otsu_threshold(gray_array: np.ndarray) -> float:
    """
    Computes Otsu's optimal threshold maximizing inter-class variance.
    """
    flat = gray_array.ravel()
    flat = flat[~np.isnan(flat)]
    if len(flat) == 0:
        return 0.5

    min_v, max_v = float(flat.min()), float(flat.max())
    if max_v - min_v < 1e-6:
        return float(min_v)

    scaled = np.uint8(np.clip((flat - min_v) / (max_v - min_v) * 255.0, 0, 255))
    hist, _ = np.histogram(scaled, bins=256, range=(0, 256))
    hist = hist.astype(np.float64) / len(flat)

    weight1 = np.cumsum(hist)
    weight2 = 1.0 - weight1

    cum_mean = np.cumsum(np.arange(256) * hist)
    total_mean = cum_mean[-1]

    mean1 = cum_mean / (weight1 + 1e-12)
    mean2 = (total_mean - cum_mean) / (weight2 + 1e-12)

    variance = weight1 * weight2 * ((mean1 - mean2) ** 2)
    variance[weight1 == 0] = 0.0
    variance[weight2 == 0] = 0.0

    max_val = np.max(variance)
    if max_val <= 0:
        return float((min_v + max_v) / 2.0)

    best_bins = np.where(variance >= max_val - 1e-7)[0]
    best_bin = int(np.mean(best_bins))
    return float(min_v + (best_bin / 255.0) * (max_v - min_v))


class AttentionLocalization:
    """
    Endogenous Attention Localization and Mask Refinement.
    """
    def __init__(self, smoothing_kernel_size: int = 5, top_k_pairs: int = 4):
        self.smoothing_kernel_size = smoothing_kernel_size
        self.top_k_pairs = top_k_pairs
        self.captured_attentions: Dict[str, torch.Tensor] = {}

    def clear(self):
        self.captured_attentions.clear()

    def register_hook(self, module: torch.nn.Module, layer_name: str):
        """
        Registers forward hook on a DiT attention block to capture image-to-text attention.
        """
        def hook(mod, inputs, outputs):
            # If attention weights are returned or accessible
            if isinstance(outputs, tuple) and len(outputs) > 1:
                attn_weights = outputs[1]
                if attn_weights is not None:
                    self.captured_attentions[layer_name] = attn_weights.detach().cpu()
        return module.register_forward_hook(hook)

    def refine_attention_map(
        self,
        attn_map: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Performs topology-aware refinement on aggregated attention map:
        1. Local neighborhood smoothing
        2. Otsu thresholding
        3. Upscaling/downscaling to target latent resolution

        :param attn_map: [H_attn, W_attn] or [1, 1, H_attn, W_attn]
        :param target_h: Target latent height
        :param target_w: Target latent width
        :return: Binary/Soft mask tensor of shape [1, 1, target_h, target_w]
        """
        if attn_map.ndim == 2:
            attn_map = attn_map.unsqueeze(0).unsqueeze(0)
        elif attn_map.ndim == 3:
            attn_map = attn_map.unsqueeze(0)

        # 1. Spatial smoothing via Gaussian-like average pooling
        k = self.smoothing_kernel_size
        pad = k // 2
        smoothed = F.avg_pool2d(attn_map, kernel_size=k, stride=1, padding=pad)

        # Normalize to [0, 1]
        min_v = smoothed.min()
        max_v = smoothed.max()
        if max_v - min_v > 1e-6:
            normalized = (smoothed - min_v) / (max_v - min_v)
        else:
            normalized = smoothed

        # 2. Otsu thresholding
        np_arr = normalized.squeeze().cpu().numpy()
        thresh = otsu_threshold(np_arr)
        binary_np = (np_arr >= thresh).astype(np.float32)

        # 3. Resize to target latent resolution
        binary_tensor = torch.from_numpy(binary_np).unsqueeze(0).unsqueeze(0).to(
            device=attn_map.device, dtype=attn_map.dtype
        )
        mask = F.interpolate(binary_tensor, size=(target_h, target_w), mode="bilinear", align_corners=False)
        mask = (mask > 0.3).to(dtype=attn_map.dtype)

        return mask

    def create_layout_mask(
        self,
        regions: List[Dict],
        latent_h: int,
        latent_w: int,
        img_w: int = 1024,
        img_h: int = 1024,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Builds a binary mask in latent resolution from parsed text regions.
        """
        mask = torch.zeros((1, 1, latent_h, latent_w), device=device, dtype=dtype)
        scale_x = latent_w / img_w
        scale_y = latent_h / img_h

        for reg in regions:
            x1, y1, x2, y2 = reg["box"]
            lx1 = max(0, int(x1 * scale_x))
            ly1 = max(0, int(y1 * scale_y))
            lx2 = min(latent_w, int(x2 * scale_x) + 1)
            ly2 = min(latent_h, int(y2 * scale_y) + 1)

            mask[:, :, ly1:ly2, lx1:lx2] = 1.0

        return mask
