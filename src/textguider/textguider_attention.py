"""
TextGuider Attention Extraction for FLUX.2 Klein 4B.
arXiv:2512.09350

Gradient-enabled attention capture for computing TextGuider losses.
Unlike the existing attention_capture.py (which uses @torch.no_grad for
inference-only attribution), this module retains gradients so that the
latent can be updated via ∇_{Z_{t_k}} L.

Key design choices following the paper:
  - Section B: "We compute the latent gradients using all 19 layers of
    dual-stream blocks" → For Klein 4B, we use all 5 dual-stream blocks.
  - Section 3.1: A = softmax(Q_img @ K_text^T / √d) — image queries,
    text keys (NOT the reverse direction A^rev used by AMO Sampler).
  - Section A.2: This direction gives better token-level spatial separation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class TextGuiderAttentionStore:
    """Accumulates cross-modal attention maps across dual-stream layers.

    Stores A_{τ} for selected text token indices, where:
      A = softmax(Q_img @ K_text^T / √d)
      A_{τ} = column of A corresponding to text token τ

    The attention maps retain their computation graph for gradient backprop.
    """

    def __init__(
        self,
        quo_indices: List[int],
        text_token_indices: List[List[int]],
        num_heads: int,
    ):
        self.quo_indices = quo_indices  # Token indices of opening quotes
        self.text_token_indices = text_token_indices  # Per-span token indices
        self.num_heads = num_heads

        # Per-layer storage: list of (attn_map_quo, [attn_map_text_i, ...])
        self.layer_attentions: List[Tuple[Tensor, List[Tensor]]] = []

    def clear(self):
        """Clear stored attention maps for a new denoising step."""
        self.layer_attentions.clear()

    def store_attention(
        self,
        q_img: Tensor,
        k_text: Tensor,
        text_len: int,
    ) -> None:
        """Compute and store cross-modal attention A = softmax(Q_img @ K_text^T / √d).

        Args:
            q_img: Image query tensor, shape [B, H, N_img, D] (after RoPE).
            k_text: Text key tensor, shape [B, H, N_text, D] (after RoPE).
            text_len: Number of text tokens.
        """
        B, H, N_img, D = q_img.shape
        scale = 1.0 / math.sqrt(D)

        # Compute attention scores: [B, H, N_img, N_text]
        attn_logits = torch.matmul(q_img, k_text.transpose(-1, -2)) * scale
        # Softmax over text dimension (keys)
        attn_probs = F.softmax(attn_logits, dim=-1)

        # Average over heads: [B, N_img, N_text]
        attn_mean = attn_probs.mean(dim=1)

        # Extract attention maps for quotation mark tokens
        # Average over all quotation mark tokens if multiple
        if self.quo_indices:
            valid_quo = [idx for idx in self.quo_indices if idx < text_len]
            if valid_quo:
                quo_cols = attn_mean[:, :, valid_quo]  # [B, N_img, num_quo]
                attn_quo = quo_cols.mean(dim=-1)  # [B, N_img]
            else:
                attn_quo = torch.zeros(B, N_img, device=q_img.device, dtype=q_img.dtype)
        else:
            attn_quo = torch.zeros(B, N_img, device=q_img.device, dtype=q_img.dtype)

        # Extract attention maps for each textual content token tau_i
        attn_texts = []
        # Flatten token groups to individual token indices
        flat_indices = []
        for group in self.text_token_indices:
            if isinstance(group, list):
                flat_indices.extend(group)
            else:
                flat_indices.append(group)

        for idx in flat_indices:
            if idx < text_len:
                attn_text = attn_mean[:, :, idx]  # [B, N_img]
            else:
                attn_text = torch.zeros(B, N_img, device=q_img.device, dtype=q_img.dtype)
            attn_texts.append(attn_text)

        self.layer_attentions.append((attn_quo, attn_texts))

    def get_aggregated_maps(self) -> Tuple[Tensor, List[Tensor]]:
        """Aggregate attention maps across all dual-stream layers.

        Returns the average attention maps over all layers, preserving
        the computation graph for gradient backpropagation.

        Returns:
            (attn_quo, attn_texts) where:
              attn_quo: [B, N_img] — quotation mark attention
              attn_texts: List[[B, N_img]] — per-token text attention
        """
        if not self.layer_attentions:
            raise RuntimeError("No attention maps captured. Run model forward first.")

        # Average quotation mark attention across layers
        quo_maps = [la[0] for la in self.layer_attentions]
        avg_quo = torch.stack(quo_maps, dim=0).mean(dim=0)  # [B, N_img]

        # Average each text token attention across layers
        num_tokens = len(self.layer_attentions[0][1])
        avg_texts = []
        for t in range(num_tokens):
            token_maps = [la[1][t] for la in self.layer_attentions]
            avg_text = torch.stack(token_maps, dim=0).mean(dim=0)  # [B, N_img]
            avg_texts.append(avg_text)

        return avg_quo, avg_texts


class DoubleBlockAttentionHook:
    """Forward hook for FLUX.2 Klein's DoubleStreamBlock to capture Q_img, K_text.

    This hook intercepts the attention computation in dual-stream blocks
    to extract the cross-modal attention map A = softmax(Q_img @ K_text^T / √d).

    For FLUX.2 Klein 4B, dual-stream blocks compute attention as:
      q = cat(txt_q, img_q); k = cat(txt_k, img_k); v = cat(txt_v, img_v)
    After RoPE, the full attention is computed jointly.
    We extract Q_img and K_text columns from the concatenated tensors.
    """

    def __init__(self, store: TextGuiderAttentionStore, block_idx: int):
        self.store = store
        self.block_idx = block_idx
        self._handle = None

    def _hook_fn(self, module, args, output):
        """Hook invoked after DoubleStreamBlock.forward_kv_extract or forward.

        For TextGuider, we need to intercept BEFORE the attention is computed
        to access Q and K with gradients. We achieve this by hooking into
        the _prepare_qkv output.
        """
        # This hook is registered differently - see TextGuiderForwardWrapper
        pass


class TextGuiderForwardWrapper:
    """Wraps the FLUX.2 Klein model forward pass for TextGuider gradient computation.

    Performs a forward pass through dual-stream blocks with gradient tracking,
    extracts cross-modal attention maps, computes the TextGuider loss, and
    returns the gradient ∇_{Z_{t_k}} L.

    Paper Section B:
      "We compute the latent gradients using all 19 layers of dual-stream
       blocks, while excluding the single-stream blocks from gradient
       computation."

    For FLUX.2 Klein 4B: 5 dual-stream blocks, 20 single-stream blocks.
    """

    def __init__(
        self,
        model,
        store: TextGuiderAttentionStore,
        use_gradient_checkpointing: bool = True,
    ):
        self.model = model
        self.store = store
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def compute_attention_maps_diffusers(
        self,
        transformer,
        latents: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_ids: Optional[Tensor] = None,
        txt_ids: Optional[Tensor] = None,
        guidance: Optional[Tensor] = None,
        joint_attention_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Compute TextGuider attention maps through a Diffusers Flux2Transformer."""
        self.store.clear()

        return self._compute_via_processor_hooks(
            transformer=transformer,
            latents=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            **kwargs,
        )

    def _compute_via_processor_hooks(
        self,
        transformer,
        latents: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        img_ids: Optional[Tensor] = None,
        txt_ids: Optional[Tensor] = None,
        guidance: Optional[Tensor] = None,
        joint_attention_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Compute attention maps by hooking into the transformer's attention layers."""
        captured_qk: List[Tuple[Tensor, Tensor, int]] = []
        original_processors = {}

        try:
            for name, module in transformer.named_modules():
                class_name = module.__class__.__name__
                if class_name in {"Flux2Attention", "Flux2ParallelSelfAttention", "FluxAttention"}:
                    if "transformer_blocks." in name and "single_" not in name:
                        if hasattr(module, "get_processor"):
                            original_processors[name] = module.get_processor()
                        elif hasattr(module, "processor"):
                            original_processors[name] = module.processor

                        processor = _GradientCaptureProcessor(
                            captured_qk, self.store, name
                        )
                        if hasattr(module, "set_processor"):
                            module.set_processor(processor)

            # Build forward arguments dynamically based on signature
            import inspect
            sig = inspect.signature(transformer.forward)
            params = sig.parameters

            fwd_kwargs = {"return_dict": False}
            if "hidden_states" in params:
                fwd_kwargs["hidden_states"] = latents
            if "encoder_hidden_states" in params and encoder_hidden_states is not None:
                fwd_kwargs["encoder_hidden_states"] = encoder_hidden_states
            if "timestep" in params and timestep is not None:
                fwd_kwargs["timestep"] = timestep
            if "img_ids" in params and img_ids is not None:
                fwd_kwargs["img_ids"] = img_ids
            if "txt_ids" in params and txt_ids is not None:
                fwd_kwargs["txt_ids"] = txt_ids
            if "guidance" in params and guidance is not None:
                fwd_kwargs["guidance"] = guidance
            if "joint_attention_kwargs" in params and joint_attention_kwargs is not None:
                fwd_kwargs["joint_attention_kwargs"] = joint_attention_kwargs
            if "pooled_projections" in params and "pooled_projections" in kwargs:
                fwd_kwargs["pooled_projections"] = kwargs["pooled_projections"]

            # Forward pass through the transformer with gradient tracking
            transformer_output = transformer(**fwd_kwargs)

        finally:
            # Restore original processors
            for name, processor in original_processors.items():
                module = dict(transformer.named_modules()).get(name)
                if module is not None and hasattr(module, "set_processor"):
                    module.set_processor(processor)

        # Aggregate attention maps across layers
        return self.store.get_aggregated_maps()

    def compute_attention_maps_native(
        self,
        model,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """Compute TextGuider attention maps using the native FLUX.2 model.

        For the official FLUX.2 repository's model implementation (non-Diffusers).
        Performs forward pass through dual-stream blocks with gradient tracking.

        Args:
            model: Flux2 model from third_party/flux2-official.
            img: Image tokens [B, N_img, C].
            img_ids: Image position IDs [B, N_img, 4].
            txt: Text tokens [B, N_txt, C_txt].
            txt_ids: Text position IDs [B, N_txt, 4].
            timesteps: Timestep tensor [B].
            guidance: Guidance tensor [B] or None.

        Returns:
            (velocity_pred, attn_quo, attn_texts)
        """
        from third_party.flux2_official.src.flux2.model import (
            timestep_embedding,
            apply_rope,
        )

        self.store.clear()

        num_txt_tokens = txt.shape[1]

        # Compute embeddings
        timestep_emb = timestep_embedding(timesteps, 256)
        vec = model.time_in(timestep_emb)
        if model.use_guidance_embed and guidance is not None:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + model.guidance_in(guidance_emb)

        double_block_mod_img = model.double_stream_modulation_img(vec)
        double_block_mod_txt = model.double_stream_modulation_txt(vec)
        single_block_mod, _ = model.single_stream_modulation(vec)

        img_emb = model.img_in(img)
        txt_emb = model.txt_in(txt)

        pe_x = model.pe_embedder(img_ids)
        pe_ctx = model.pe_embedder(txt_ids)

        # Forward through dual-stream blocks WITH gradient tracking
        for block_idx, block in enumerate(model.double_blocks):
            # Prepare QKV
            q, k, v, pe_full, n_txt, mods = block._prepare_qkv(
                img_emb, txt_emb, pe_x, pe_ctx,
                double_block_mod_img, double_block_mod_txt,
            )
            q, k = apply_rope(q, k, pe_full)

            # Extract Q_img and K_text for TextGuider
            # q layout: [txt_q, img_q], k layout: [txt_k, img_k]
            q_img = q[:, :, n_txt:, :]  # [B, H, N_img, D]
            k_text = k[:, :, :n_txt, :]  # [B, H, N_txt, D]

            self.store.store_attention(q_img, k_text, n_txt)

            # Continue with standard attention and residuals
            from torch.nn.functional import scaled_dot_product_attention
            attn = scaled_dot_product_attention(q, k, v, is_causal=False)
            from einops import rearrange
            attn = rearrange(attn, "b h n d -> b n (h d)")

            txt_attn = attn[:, :n_txt]
            img_attn = attn[:, n_txt:]
            img_emb, txt_emb = block._apply_residuals(
                img_emb, txt_emb, img_attn, txt_attn, mods
            )

        # Single-stream blocks (no gradient needed for TextGuider)
        with torch.no_grad():
            img_combined = torch.cat((txt_emb, img_emb), dim=1)
            pe = torch.cat((pe_ctx, pe_x), dim=2)
            for block in model.single_blocks:
                img_combined, _ = block.forward_kv_extract(
                    img_combined, pe, single_block_mod, num_txt_tokens, num_ref_tokens=0,
                )
            img_out = img_combined[:, num_txt_tokens:, ...]

        img_out = model.final_layer(img_out, vec)

        # Get aggregated attention maps
        attn_quo, attn_texts = self.store.get_aggregated_maps()

        return img_out, attn_quo, attn_texts


class _GradientCaptureProcessor:
    """Attention processor that captures Q_img and K_text with gradients.

    Drop-in replacement for Flux2AttnProcessor that additionally stores
    cross-modal attention maps for TextGuider loss computation.
    Gradients are retained (no detach/no_grad).
    """

    def __init__(
        self,
        captured_qk: List,
        store: TextGuiderAttentionStore,
        layer_name: str,
    ):
        self.captured_qk = captured_qk
        self.store = store
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
    ):
        """Process attention with gradient-enabled cross-modal capture.

        Follows the same computation as Flux2AttnProcessor but additionally
        captures Q_img and K_text for TextGuider.
        """
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

        # --- TextGuider: capture Q_img and K_text WITH gradients ---
        if text_len > 0:
            # Reshape to [B, H, L, D] for attention computation
            # Diffusers uses [B, L, H, D] convention
            q_bhld = query.permute(0, 2, 1, 3)  # [B, H, L, D]
            k_bhld = key.permute(0, 2, 1, 3)  # [B, H, L, D]

            q_img = q_bhld[:, :, text_len:, :]  # [B, H, N_img, D]
            k_text = k_bhld[:, :, :text_len, :]  # [B, H, N_text, D]

            self.store.store_attention(q_img, k_text, text_len)

        # Standard attention computation (same as base processor)
        # Try to use the base processor's attention dispatch
        try:
            from diffusers.models.transformers.transformer_flux2 import Flux2AttnProcessor
            _base_cls = Flux2AttnProcessor
        except ImportError:
            from diffusers.models.transformers.transformer_flux2 import Flux2ParallelSelfAttnProcessor
            _base_cls = Flux2ParallelSelfAttnProcessor

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=getattr(_base_cls, "_attention_backend", None),
            parallel_config=getattr(_base_cls, "_parallel_config", None),
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states_out, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
        else:
            encoder_hidden_states_out = None

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states_out is not None:
            return (hidden_states, encoder_hidden_states_out)
        return hidden_states
