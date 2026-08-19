"""Inference-only attention capture for Diffusers FLUX.2 Klein.

The stock Flux2 attention processor uses an optimized SDPA kernel and does not
return probabilities. This processor preserves that output path and computes a
small, detached image-to-text attribution map for FreeText localization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


class Flux2AttentionRecorder:
    """Collects per-layer image-to-text maps between pipeline callbacks."""

    def __init__(
        self,
        target_groups: list[list[int]],
        sink_indices: list[int],
        text_len: int = 0,
        image_tokens: int = 0,
        max_query_chunk: int = 256,
    ):
        self.target_groups = [list(dict.fromkeys(g)) for g in target_groups]
        self.sink_indices = list(dict.fromkeys(sink_indices))
        self.text_len = int(text_len)
        self.image_tokens = int(image_tokens)
        self.max_query_chunk = max_query_chunk
        self.pending: Dict[str, Dict[int, torch.Tensor]] = {}
        self.records: list[dict[str, Any]] = []

    def clear_pending(self) -> None:
        self.pending.clear()

    def finalize_step(self, step: int) -> None:
        for layer_name, maps in self.pending.items():
            for target_index, attn_map in maps.items():
                self.records.append({"step": int(step), "layer": layer_name, "target": target_index, "map": attn_map})
        self.clear_pending()

    @torch.no_grad()
    def capture(self, layer_name: str, query_img: torch.Tensor, key_all: torch.Tensor, text_len: int) -> None:
        """Capture mean-head attention for image queries and selected text keys.

        Tensors use Diffusers' [batch, sequence, heads, head_dim] convention.
        Only selected token columns are materialized and moved to CPU.
        """
        if not self.target_groups or query_img.ndim != 4 or key_all.ndim != 4:
            return
        batch, query_len, heads, head_dim = query_img.shape
        if self.image_tokens > 0:
            query_img = query_img[:, : self.image_tokens]
            query_len = query_img.shape[1]
        key_len = min(int(text_len), key_all.shape[1])
        all_selected = set(self.sink_indices)
        for group in self.target_groups:
            all_selected.update(i for i in group if 0 <= i < key_len)
        selected = sorted(i for i in all_selected if 0 <= i < key_len)
        if not selected:
            return

        q = query_img.float().permute(0, 2, 1, 3)  # B,H,Q,D
        k = key_all.float().permute(0, 2, 1, 3)   # B,H,T+I,D
        selected_tensor = torch.tensor(selected, device=k.device, dtype=torch.long)
        output: Dict[int, torch.Tensor] = {}
        scale = 1.0 / math.sqrt(head_dim)

        # Compute log-softmax in query chunks. This avoids allocating the full
        # Q_image x (Q_image + Q_text) attention matrix.
        for start in range(0, query_len, self.max_query_chunk):
            end = min(start + self.max_query_chunk, query_len)
            logits = torch.matmul(q[:, :, start:end, :], k.transpose(-1, -2)) * scale
            # Normalize over the complete joint sequence, then read only the
            # text-key columns. This is the image-to-text attribution in the
            # paper, not a text-only renormalization.
            probs = logits.softmax(dim=-1)[..., :key_len].index_select(-1, selected_tensor)
            for target_index, group in enumerate(self.target_groups):
                indices = [selected.index(i) for i in [*group, *self.sink_indices] if i in selected]
                if not indices:
                    continue
                value = probs[..., indices].mean(dim=(1, 3))  # B,Q_chunk
                output.setdefault(target_index, []).append(value.cpu())

        self.pending[layer_name] = {k: torch.cat(v, dim=1) for k, v in output.items()}


class Flux2CaptureProcessor:
    """Drop-in replacement for Diffusers' Flux2AttnProcessor."""

    def __init__(self, recorder: Flux2AttentionRecorder, layer_name: str):
        self.recorder = recorder
        self.layer_name = layer_name
        try:
            from diffusers.models.transformers.transformer_flux2 import Flux2ParallelSelfAttnProcessor
            self._base_cls = Flux2ParallelSelfAttnProcessor
        except ImportError as exc:  # pragma: no cover - exercised on server only
            raise ImportError("This capture processor requires a Diffusers Flux2 implementation") from exc
        try:
            from diffusers.models.transformers.transformer_flux2 import (
                Flux2AttnProcessor,
            )
            self._base_cls = Flux2AttnProcessor
        except ImportError as exc:  # pragma: no cover - exercised on server only
            raise ImportError("This capture processor requires a Diffusers Flux2 implementation") from exc

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        from diffusers.models.transformers.transformer_flux2 import (
            _get_qkv_projections,
            apply_rotary_emb,
            dispatch_attention_fn,
        )

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        text_len = 0
        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
            text_len = encoder_hidden_states.shape[1]

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if text_len:
            self.recorder.capture(self.layer_name, query[:, text_len:], key, text_len)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=getattr(self._base_cls, "_attention_backend", None),
            parallel_config=getattr(self._base_cls, "_parallel_config", None),
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return (hidden_states, encoder_hidden_states) if encoder_hidden_states is not None else hidden_states


class Flux2SingleCaptureProcessor:
    """Capture image-to-text maps from Klein's fused single-stream blocks."""

    def __init__(self, recorder: Flux2AttentionRecorder, layer_name: str):
        self.recorder = recorder
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
        from diffusers.models.transformers.transformer_flux2 import apply_rotary_emb, dispatch_attention_fn

        projected = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            projected, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        text_len = min(self.recorder.text_len, hidden_states.shape[1])
        if text_len:
            self.recorder.capture(self.layer_name, query[:, text_len:], key, text_len)
        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=getattr(self._base_cls, "_attention_backend", None),
            parallel_config=getattr(self._base_cls, "_parallel_config", None),
        ).flatten(2, 3).to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        return attn.to_out(torch.cat([attn_output, mlp_hidden_states], dim=-1))


class Flux2CaptureHandle:
    def __init__(self, originals: dict[str, Any], recorder: Flux2AttentionRecorder):
        self.originals = originals
        self.recorder = recorder

    def close(self, transformer) -> None:
        for name, processor in self.originals.items():
            module = dict(transformer.named_modules()).get(name)
            if module is not None and hasattr(module, "set_processor"):
                module.set_processor(processor)


def install_flux2_capture(transformer, recorder: Flux2AttentionRecorder) -> Flux2CaptureHandle:
    originals = {}
    for name, module in transformer.named_modules():
        class_name = module.__class__.__name__
        if class_name in {"Flux2Attention", "Flux2ParallelSelfAttention"} and hasattr(module, "set_processor"):
            originals[name] = module.get_processor() if hasattr(module, "get_processor") else module.processor
            processor = (
                Flux2CaptureProcessor(recorder, name)
                if class_name == "Flux2Attention"
                else Flux2SingleCaptureProcessor(recorder, name)
            )
            module.set_processor(processor)
    if not originals:
        raise RuntimeError("No Flux2Attention modules found; unsupported Diffusers transformer")
    return Flux2CaptureHandle(originals, recorder)
