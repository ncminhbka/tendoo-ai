"""
TextGuider Attention Extraction for FLUX.2 Klein 4B.
arXiv:2512.09350

Gradient-enabled attention capture để tính TextGuider loss.

Cập nhật quan trọng (xem ARCHITECTURE_NOTES.md):
  - KHÔNG còn hardcode số layer double/single-stream — đọc động từ
    `transformer.config.num_layers` / `num_single_layers` (đã xác nhận qua
    tài liệu Diffusers `Flux2Transformer2DModel`).
  - Nhận diện double-stream block theo CẤU TRÚC (`hasattr(module,
    "to_added_qkv")`) thay vì đoán tên module/class theo quy ước của
    FLUX.1 — vì tên class Attention Processor thật của Flux2 trong
    Diffusers chưa được xác nhận công khai.
  - Bỏ hoàn toàn `pooled_projections` — `Flux2Transformer2DModel.forward`
    KHÔNG có tham số này (khác FLUX.1/SD3).
  - Xoá `DoubleBlockAttentionHook` (dead code, chưa từng được dùng).
  - Thêm chế độ nghiêm ngặt: nếu không capture được attention map nào,
    raise lỗi rõ ràng thay vì âm thầm trả về loss=0.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class AttentionCaptureError(RuntimeError):
    """Raised khi cơ chế hook không capture được attention map nào.

    Ở bản cũ, mọi lỗi hook (sai tên module, sai attribute...) đều bị nuốt
    bởi `except Exception` ở tầng pipeline và fallback về latent gốc —
    khiến TextGuider trông như đang chạy trong khi thực ra không làm gì.
    Lớp lỗi này để phát hiện sớm tình huống đó khi strict_mode=True.
    """


class TextGuiderAttentionStore:
    """Tích luỹ attention map liên-modal qua các layer double-stream.

    Lưu A_{tau} cho các token index được chọn, với:
      A = softmax(Q_img @ K_text^T / sqrt(d))
    Giữ nguyên computation graph để backprop.
    """

    def __init__(
        self,
        quo_indices: List[int],
        text_token_indices: List[List[int]],
        num_heads: Optional[int] = None,
    ):
        self.quo_indices = quo_indices
        self.text_token_indices = text_token_indices
        # Kept for compatibility with older callers. Attention tensors already
        # carry their head dimension, so this value is intentionally unused.
        self.num_heads = num_heads

        self.layer_attentions: List[Tuple[Tensor, List[Tensor]]] = []

    def clear(self):
        self.layer_attentions.clear()

    def store_attention(
        self,
        q_img: Tensor,
        k_text: Tensor,
        text_len: int,
    ) -> None:
        """Tính và lưu A = softmax(Q_img @ K_text^T / sqrt(d)).

        Args:
            q_img: [B, H, N_img, D] (sau RoPE).
            k_text: [B, H, N_text, D] (sau RoPE).
            text_len: số token văn bản thật trong chuỗi này.
        """
        B, H, N_img, D = q_img.shape
        scale = 1.0 / math.sqrt(D)

        attn_logits = torch.matmul(q_img, k_text.transpose(-1, -2)) * scale
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_mean = attn_probs.mean(dim=1)  # average over heads: [B, N_img, N_text]

        if self.quo_indices:
            valid_quo = [idx for idx in self.quo_indices if idx < text_len]
            if valid_quo:
                quo_cols = attn_mean[:, :, valid_quo]
                attn_quo = quo_cols.mean(dim=-1)
            else:
                attn_quo = torch.zeros(B, N_img, device=q_img.device, dtype=q_img.dtype)
        else:
            attn_quo = torch.zeros(B, N_img, device=q_img.device, dtype=q_img.dtype)

        flat_indices = []
        for group in self.text_token_indices:
            if isinstance(group, list):
                flat_indices.extend(group)
            else:
                flat_indices.append(group)

        attn_texts = []
        for idx in flat_indices:
            if idx < text_len:
                attn_text = attn_mean[:, :, idx]
            else:
                attn_text = torch.zeros(B, N_img, device=q_img.device, dtype=q_img.dtype)
            attn_texts.append(attn_text)

        self.layer_attentions.append((attn_quo, attn_texts))

    def get_aggregated_maps(self, strict: bool = True) -> Tuple[Tensor, List[Tensor]]:
        """Trung bình attention map qua các layer double-stream đã capture."""
        if not self.layer_attentions:
            if strict:
                raise AttentionCaptureError(
                    "[TextGuider] Không capture được attention map nào. "
                    "Cơ chế hook (theo tên module hoặc theo attribute "
                    "to_added_qkv) không khớp với kiến trúc transformer "
                    "thật — kiểm tra lại transformer.named_modules() để "
                    "xác nhận đúng block double-stream."
                )
            device = "cpu"
            return torch.zeros(1, 1, device=device), []

        quo_maps = [la[0] for la in self.layer_attentions]
        avg_quo = torch.stack(quo_maps, dim=0).mean(dim=0)

        num_tokens = len(self.layer_attentions[0][1])
        avg_texts = []
        for t in range(num_tokens):
            token_maps = [la[1][t] for la in self.layer_attentions]
            avg_text = torch.stack(token_maps, dim=0).mean(dim=0)
            avg_texts.append(avg_text)

        return avg_quo, avg_texts


def _is_double_stream_attention(module) -> bool:
    """Nhận diện attention module thuộc block double-stream theo CẤU TRÚC.

    Theo mô tả Diffusers cho Flux2 (`attn.to_added_qkv(encoder_hidden_states)`),
    chỉ double-stream block mới có phép chiếu QKV riêng cho token văn bản
    (`to_added_qkv`) — single-stream block gộp QKV+MLP chung một phép
    chiếu (fused). Cách này bền hơn việc đoán tên class/module.
    """
    # Diffusers may leave the text projections unfused (add_q_proj/add_k_proj/
    # add_v_proj) or fuse them into to_added_qkv. Both represent the
    # double-stream Flux2 attention; single-stream attention has neither.
    has_added_projections = (
        getattr(module, "added_kv_proj_dim", None) is not None
        and hasattr(module, "add_q_proj")
        and hasattr(module, "add_k_proj")
        and hasattr(module, "add_v_proj")
    )
    return hasattr(module, "to_q") and (
        hasattr(module, "to_added_qkv") or has_added_projections
    )


def build_guidance_tensor(
    guidance_scale: float,
    batch_size: int,
    device,
    dtype,
) -> Tensor:
    """Tạo tensor guidance hợp lệ cho transformer.

    Flux2Transformer2DModel có guidance_embeds=True theo mặc định (xác
    nhận qua tài liệu Diffusers) => KHÔNG được truyền guidance=None, kể cả
    khi chỉ dùng để capture attention (không phải lượt sinh ảnh cuối).
    """
    return torch.full((batch_size,), float(guidance_scale), device=device, dtype=dtype)


class TextGuiderForwardWrapper:
    """Wrap forward pass của FLUX.2 Klein để tính gradient cho TextGuider.

    Paper Section B: chỉ backprop qua các layer double-stream, loại trừ
    single-stream khỏi gradient. Bản này áp dụng NHẤT QUÁN cho cả đường
    Diffusers lẫn đường native (bản cũ chỉ áp đúng ở đường native, đường
    Diffusers vô tình backprop qua toàn bộ transformer).
    """

    def __init__(
        self,
        model,
        store: TextGuiderAttentionStore,
        use_gradient_checkpointing: bool = True,
        strict_mode: bool = True,
    ):
        self.model = model
        self.store = store
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.strict_mode = strict_mode

    # ------------------------------------------------------------------
    # Đường Diffusers (Flux2KleinPipeline + Flux2Transformer2DModel)
    # ------------------------------------------------------------------

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
    ) -> Tuple[Tensor, List[Tensor]]:
        """Tính attention map bằng cách hook vào các attention layer double-stream.

        LƯU Ý: hàm này chỉ cần chạy tới hết các block double-stream là đủ
        để có attention map — nhưng vì ta cũng cần một `noise_pred` hợp lệ
        ở một số đường gọi, forward vẫn chạy hết transformer. Nếu chỉ cần
        attention map (không cần noise_pred), có thể dừng sớm bằng cách
        raise ngay sau khi capture đủ `num_layers` layer double-stream
        (xem `_compute_via_processor_hooks`).
        """
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
    ) -> Tuple[Tensor, List[Tensor]]:
        original_processors = {}
        num_double_layers_expected = getattr(transformer.config, "num_layers", None)
        double_layer_names: List[str] = []

        try:
            for name, module in transformer.named_modules():
                if _is_double_stream_attention(module):
                    double_layer_names.append(name)
                    if hasattr(module, "get_processor"):
                        original_processors[name] = module.get_processor()
                    elif hasattr(module, "processor"):
                        original_processors[name] = module.processor

                    processor = _GradientCaptureProcessor(self.store, name)
                    if hasattr(module, "set_processor"):
                        module.set_processor(processor)

            if not double_layer_names:
                raise AttentionCaptureError(
                    "[TextGuider] Không tìm thấy attention module nào có "
                    "`to_added_qkv` (double-stream) trong transformer. "
                    "Kiến trúc thật có thể khác giả định — hãy in "
                    "transformer.named_modules() để kiểm tra."
                )
            if num_double_layers_expected is not None and (
                len(double_layer_names) != num_double_layers_expected
            ):
                print(
                    f"[TextGuider] Cảnh báo: config báo num_layers="
                    f"{num_double_layers_expected} nhưng phát hiện "
                    f"{len(double_layer_names)} module double-stream. "
                    f"Vẫn tiếp tục nhưng nên kiểm tra lại."
                )

            fwd_kwargs = {
                "hidden_states": latents,
                "encoder_hidden_states": encoder_hidden_states,
                "timestep": timestep,
                "img_ids": img_ids,
                "txt_ids": txt_ids,
                "guidance": guidance,
                "joint_attention_kwargs": joint_attention_kwargs,
                "return_dict": False,
            }
            # KHÔNG truyền pooled_projections: Flux2Transformer2DModel.forward
            # không có tham số này (xem ARCHITECTURE_NOTES.md mục 2).

            transformer(**fwd_kwargs)

        finally:
            for name, processor in original_processors.items():
                module = dict(transformer.named_modules()).get(name)
                if module is not None and hasattr(module, "set_processor"):
                    module.set_processor(processor)

        return self.store.get_aggregated_maps(strict=self.strict_mode)

    # ------------------------------------------------------------------
    # Đường native (repo chính thức black-forest-labs/flux2)
    # ------------------------------------------------------------------

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
        """Dùng trực tiếp class `Flux2`/`DoubleStreamBlock`/`SingleStreamBlock`
        từ repo chính thức `black-forest-labs/flux2` (src/flux2/model.py).

        Số layer double/single-stream được đọc động từ chính `model` (qua
        `len(model.double_blocks)` / `len(model.single_blocks)`), không
        hardcode — vì con số cụ thể của riêng Klein 4B chưa được xác nhận
        công khai (xem ARCHITECTURE_NOTES.md mục 9).
        """
        from flux2.model import apply_rope  # type: ignore

        self.store.clear()

        num_txt_tokens = txt.shape[1]

        timestep_emb = model.time_in(model.timestep_embedding(timesteps, 256))
        vec = timestep_emb
        if getattr(model.params, "guidance_embed", False) and guidance is not None:
            guidance_emb = model.timestep_embedding(guidance, 256)
            vec = vec + model.guidance_in(guidance_emb)

        double_block_mod_img = model.double_stream_modulation_img(vec)
        double_block_mod_txt = model.double_stream_modulation_txt(vec)
        single_block_mod, _ = model.single_stream_modulation(vec)

        img_emb = model.img_in(img)
        txt_emb = model.txt_in(txt)

        pe_x = model.pe_embedder(img_ids)
        pe_ctx = model.pe_embedder(txt_ids)

        for block_idx, block in enumerate(model.double_blocks):
            q, k, v, pe_full, n_txt, mods = block._prepare_qkv(
                img_emb, txt_emb, pe_x, pe_ctx,
                double_block_mod_img, double_block_mod_txt,
            )
            q, k = apply_rope(q, k, pe_full)

            # Layout xác nhận từ repo gốc: img = concat(txt, img) — văn bản
            # đứng trước ảnh trong chuỗi ghép.
            q_img = q[:, :, n_txt:, :]
            k_text = k[:, :, :n_txt, :]

            self.store.store_attention(q_img, k_text, n_txt)

            from torch.nn.functional import scaled_dot_product_attention
            attn = scaled_dot_product_attention(q, k, v, is_causal=False)
            from einops import rearrange
            attn = rearrange(attn, "b h n d -> b n (h d)")

            txt_attn = attn[:, :n_txt]
            img_attn = attn[:, n_txt:]
            img_emb, txt_emb = block._apply_residuals(
                img_emb, txt_emb, img_attn, txt_attn, mods
            )

        # Chỉ single-stream mới bị loại khỏi gradient (đúng theo paper
        # Section B) — double-stream ở trên giữ nguyên gradient.
        with torch.no_grad():
            img_combined = torch.cat((txt_emb, img_emb), dim=1)
            pe = torch.cat((pe_ctx, pe_x), dim=2)
            for block in model.single_blocks:
                img_combined, _ = block.forward_kv_extract(
                    img_combined, pe, single_block_mod, num_txt_tokens, num_ref_tokens=0,
                )
            img_out = img_combined[:, num_txt_tokens:, ...]

        img_out = model.final_layer(img_out, vec)

        attn_quo, attn_texts = self.store.get_aggregated_maps(strict=self.strict_mode)

        return img_out, attn_quo, attn_texts


class _GradientCaptureProcessor:
    """Attention processor thay thế processor gốc để bắt Q_img/K_text có gradient.

    So với bản cũ:
      - Bỏ toàn bộ logic đoán `_attention_backend`/`_parallel_config` từ
        class Flux2AttnProcessor (không có bằng chứng API này tồn tại) —
        dùng thẳng `torch.nn.functional.scaled_dot_product_attention`,
        vốn là API ổn định, chắc chắn tồn tại, và tương đương về mặt toán
        học với những gì `dispatch_attention_fn` nội bộ của Diffusers làm.
      - Không còn phụ thuộc `_get_qkv_projections` (private API dễ vỡ theo
        version) — tự tính projection qua `attn.to_q/to_k/to_v` và
        `attn.to_added_qkv`, vốn là attribute chuẩn của mọi `Attention`
        module trong Diffusers.
    """

    def __init__(self, store: TextGuiderAttentionStore, layer_name: str):
        self.store = store
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
        **kwargs,
    ):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        if hasattr(attn, "norm_q") and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, "norm_k") and attn.norm_k is not None:
            key = attn.norm_k(key)

        text_len = 0
        if encoder_hidden_states is not None and (
            hasattr(attn, "to_added_qkv") or getattr(attn, "added_kv_proj_dim", None) is not None
        ):
            if hasattr(attn, "to_added_qkv"):
                enc_q, enc_k, enc_v = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
            else:
                enc_q = attn.add_q_proj(encoder_hidden_states)
                enc_k = attn.add_k_proj(encoder_hidden_states)
                enc_v = attn.add_v_proj(encoder_hidden_states)
            enc_q = enc_q.unflatten(-1, (attn.heads, -1))
            enc_k = enc_k.unflatten(-1, (attn.heads, -1))
            enc_v = enc_v.unflatten(-1, (attn.heads, -1))
            if hasattr(attn, "norm_added_q") and attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if hasattr(attn, "norm_added_k") and attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)
            query = torch.cat([enc_q, query], dim=1)
            key = torch.cat([enc_k, key], dim=1)
            value = torch.cat([enc_v, value], dim=1)
            text_len = encoder_hidden_states.shape[1]

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if text_len > 0:
            q_bhld = query.permute(0, 2, 1, 3)
            k_bhld = key.permute(0, 2, 1, 3)
            q_img = q_bhld[:, :, text_len:, :]
            k_text = k_bhld[:, :, :text_len, :]
            self.store.store_attention(q_img, k_text, text_len)

        q_bhld = query.permute(0, 2, 1, 3)
        k_bhld = key.permute(0, 2, 1, 3)
        v_bhld = value.permute(0, 2, 1, 3)
        hidden_states = F.scaled_dot_product_attention(
            q_bhld, k_bhld, v_bhld, attn_mask=attention_mask
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3).flatten(2, 3).to(query.dtype)

        if text_len > 0:
            encoder_hidden_states_out, hidden_states = hidden_states.split_with_sizes(
                [text_len, hidden_states.shape[1] - text_len], dim=1
            )
            if hasattr(attn, "to_add_out") and attn.to_add_out is not None:
                encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
        else:
            encoder_hidden_states_out = None

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states_out is not None:
            return (hidden_states, encoder_hidden_states_out)
        return hidden_states
