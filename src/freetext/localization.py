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
        self.attention_records: List[Dict] = []
        self.target_token_groups: List[List[int]] = []
        self.sink_token_indices: List[int] = []

    def clear(self):
        self.captured_attentions.clear()
        self.attention_records.clear()

    @staticmethod
    def _token_groups(tokenizer, prompt: str, texts: List[str]) -> Tuple[List[List[int]], List[int]]:
        """Map target spans to the token sequence actually consumed by Qwen3.

        Flux2Klein does not encode ``prompt`` as a raw tokenizer sequence. Its
        pipeline first applies the Qwen3 chat template and then pads/truncates
        to 512 positions. Attention columns therefore have to be indexed in
        that same formatted sequence; using ``tokenizer(prompt)`` silently
        shifts every target token when chat-template markers are present.
        """
        groups: List[List[int]] = []
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_start = formatted.find(prompt)
            if prompt_start < 0:
                prompt_start = 0
            encoded = tokenizer(
                formatted,
                return_offsets_mapping=True,
                return_attention_mask=True,
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            offsets = encoded["offset_mapping"]
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            if hasattr(offsets, "tolist"):
                offsets = offsets.tolist()
            if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
                offsets = offsets[0]
            if hasattr(attention_mask, "tolist"):
                attention_mask = attention_mask.tolist()
            if attention_mask and isinstance(attention_mask[0], list):
                attention_mask = attention_mask[0]
            for text in texts:
                relative_start = prompt.find(text)
                start = prompt_start + relative_start if relative_start >= 0 else -1
                end = start + len(text) if start >= 0 else -1
                groups.append([
                    i for i, (left, right) in enumerate(offsets)
                    if (not attention_mask or attention_mask[i])
                    and end > left and start < right and right > left
                ])
            special_ids = set(getattr(tokenizer, "all_special_ids", []))
            # Do not treat padded positions as sink tokens: they are present
            # in the fixed 512-length sequence but are masked by Qwen3.
            sinks = [
                i for i, token_id in enumerate(input_ids)
                if (not attention_mask or attention_mask[i]) and token_id in special_ids
            ]
        except Exception:
            full = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            if hasattr(full, "tolist"):
                full = full.tolist()
            if full and isinstance(full[0], list):
                full = full[0]
            target_ids = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
            groups = []
            for ids in target_ids:
                found = []
                for start in range(max(0, len(full) - len(ids) + 1)):
                    if full[start : start + len(ids)] == ids:
                        found = list(range(start, start + len(ids)))
                        break
                groups.append(found)
            special_ids = set(getattr(tokenizer, "all_special_ids", []))
            sinks = [i for i, token_id in enumerate(full) if token_id in special_ids]
        if not sinks:
            length = max((max(g) for g in groups if g), default=1) + 1
            sinks = [0, max(0, length - 1)]
        return groups, sinks

    def configure_attention(self, tokenizer, prompt: str, texts: List[str]) -> Tuple[List[List[int]], List[int]]:
        self.target_token_groups, self.sink_token_indices = self._token_groups(tokenizer, prompt, texts)
        return self.target_token_groups, self.sink_token_indices

    def install_flux2_capture(
        self,
        transformer,
        tokenizer,
        prompt: str,
        texts: List[str],
        image_height: int = 1024,
        image_width: int = 1024,
    ):
        from .attention_capture import Flux2AttentionRecorder, install_flux2_capture

        groups, sinks = self.configure_attention(tokenizer, prompt, texts)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(
            formatted,
            padding="max_length",
            truncation=True,
            max_length=512,
        )["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        image_tokens = (int(image_height) // 16) * (int(image_width) // 16)
        recorder = Flux2AttentionRecorder(groups, sinks, text_len=len(encoded), image_tokens=image_tokens)
        handle = install_flux2_capture(transformer, recorder)
        return handle, recorder

    def finalize_attention_step(self, recorder, step: int) -> None:
        recorder.finalize_step(step)
        self.attention_records.extend(recorder.records)
        recorder.records.clear()

    @staticmethod
    def _grid_shape(num_tokens: int, img_h: int, img_w: int) -> Tuple[int, int]:
        ratio = img_h / max(img_w, 1)
        h = max(1, int(round((num_tokens * ratio) ** 0.5)))
        while h > 1 and num_tokens % h:
            h -= 1
        return h, max(1, num_tokens // h)

    @staticmethod
    def _soft_iou(pred: torch.Tensor, ref: torch.Tensor) -> float:
        pred = pred / (pred.sum() + 1e-8)
        ref = ref / (ref.sum() + 1e-8)
        intersection = (pred * ref).sum()
        union = pred.sum() + ref.sum() - intersection
        return float((intersection / (union + 1e-8)).item())

    def _topology_mask(self, attn_map: torch.Tensor) -> torch.Tensor:
        """Otsu + DBSCAN/connected-component refinement from paper Sec. 3.1.3."""
        # Local-neighborhood aggregation is applied before thresholding. Keep
        # the unsmoothed normalized map as the score map for Eq. (5).
        score_map = attn_map.float()
        kernel = max(1, int(self.smoothing_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        smoothed = F.avg_pool2d(
            score_map[None, None], kernel_size=kernel, stride=1, padding=kernel // 2
        )[0, 0]
        normalized = score_map - score_map.min()
        normalized = normalized / (normalized.max() + 1e-8)
        smoothed = smoothed - smoothed.min()
        smoothed = smoothed / (smoothed.max() + 1e-8)
        normalized_np = normalized.cpu().numpy()
        threshold = otsu_threshold(smoothed.cpu().numpy())
        foreground = smoothed.cpu().numpy() >= threshold
        coords = np.argwhere(foreground)
        if len(coords) == 0:
            return (normalized > 0).float()
        labels = None
        try:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=2.0, min_samples=max(2, min(8, len(coords) // 100))).fit(coords).labels_
        except Exception:
            try:
                from scipy import ndimage
                labels, _ = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
                labels = labels[tuple(coords.T)] - 1
            except Exception:
                return torch.from_numpy(foreground.astype(np.float32))
        valid_labels = sorted(set(int(x) for x in labels if x >= 0))
        if not valid_labels:
            return torch.from_numpy(foreground.astype(np.float32))

        # Eq. (5): tau is a high quantile over the union of candidate
        # regions, and each cluster is scored by the fraction of its points
        # above tau in the original (pre-smoothing) attention map.
        candidate_values = normalized_np[coords[:, 0], coords[:, 1]]
        tau = float(np.quantile(candidate_values, 0.90))
        best_label, best_score = None, -1.0
        for label in valid_labels:
            points = coords[labels == label]
            score = float(
                np.mean(normalized_np[points[:, 0], points[:, 1]] > tau)
            )
            if score > best_score:
                best_label, best_score = label, score
        if best_label is None:
            return torch.from_numpy(foreground.astype(np.float32))
        selected = np.zeros_like(foreground, dtype=np.float32)
        selected[coords[labels == best_label, 0], coords[labels == best_label, 1]] = 1.0
        return torch.from_numpy(selected)

    def build_attention_mask(
        self,
        regions: List[Dict],
        target_h: int,
        target_w: int,
        img_w: int,
        img_h: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """Select top-K timestep/layer maps and return a latent-space mask."""
        if not self.attention_records or not regions:
            return None
        grouped: Dict[int, List[Dict]] = {}
        for record in self.attention_records:
            grouped.setdefault(int(record["target"]), []).append(record)
        result = torch.zeros((1, 1, target_h, target_w), dtype=torch.float32)
        for target_index, records in grouped.items():
            region = regions[min(target_index, len(regions) - 1)]
            sample = records[0]["map"][0]
            qh, qw = self._grid_shape(sample.numel(), img_h, img_w)
            ref = torch.zeros((1, 1, img_h, img_w), dtype=torch.float32)
            x1, y1, x2, y2 = region["box"]
            ref[:, :, max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)] = 1.0
            ref = F.interpolate(ref, size=(qh, qw), mode="bilinear", align_corners=False).squeeze()
            scored = []
            for record in records:
                current = record["map"][0].view(qh, qw).float()
                scored.append((self._soft_iou(current, ref), current))
            selected = [current for _, current in sorted(scored, key=lambda x: x[0], reverse=True)[: self.top_k_pairs]]
            aggregate = torch.stack(selected).mean(dim=0)
            refined = self._topology_mask(aggregate)
            refined = F.interpolate(refined[None, None], size=(target_h, target_w), mode="bilinear", align_corners=False)
            result = torch.maximum(result, (refined > 0.3).float())
        return result.to(device=device, dtype=dtype)

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
