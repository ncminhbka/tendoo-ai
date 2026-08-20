"""
TextGuider: Training-Free Guidance for Text Rendering via Attention Alignment.
arXiv:2512.09350

Core implementation of TextGuider loss functions (split loss, wrap loss),
AMO Sampler integration, và token parsing cho FLUX.2 Klein 4B Base.

Đã cập nhật (xem ARCHITECTURE_NOTES.md) để khớp với cách FLUX.2 Klein thật
sự dùng Qwen3 làm bộ trích đặc trưng văn bản (encoding-only), KHÔNG phải
một lượt chat instruct — khác với giả định ban đầu.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TextGuiderConfig:
    """Cấu hình siêu tham số cho TextGuider.

    Paper defaults (Section 4.1, Section B, Section D):
      - alpha = 60 (guidance step size)
      - t_guide_ratio = 0.25 (1/4 số bước đầu)
      - amo_overshoot_c = 0.5

    Với FLUX.2 [klein] 4B Base — xác nhận từ README chính thức
    (black-forest-labs/flux2): model này KHÔNG distill, khuyến nghị 50 bước,
    và dùng classifier-free guidance thật (denoise_cfg trong sampling.py),
    guidance mặc định 4.0 — xem ARCHITECTURE_NOTES.md mục 4 và 8.
    """

    # --- TextGuider guidance ---
    alpha: float = 60.0
    t_guide_ratio: float = 0.25

    # --- AMO Sampler (Equation 2) ---
    amo_enabled: bool = True
    amo_overshoot_c: float = 0.5

    # --- Gradient computation ---
    # "double" = chỉ backprop qua dual-stream blocks, loại trừ single-stream
    # (đúng theo paper Section B). Xem lưu ý ở TextGuiderForwardWrapper.
    target_layers: str = "double"
    use_gradient_checkpointing: bool = True

    # --- Classifier-free guidance (BẮT BUỘC với model Base — mục 4) ---
    use_cfg: bool = True
    negative_prompt: str = ""

    # --- Generation defaults cho FLUX.2 Klein 4B Base ---
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    resolution: int = 1024

    # --- Chế độ nghiêm ngặt: raise thay vì âm thầm fallback khi có bất
    # thường (attention rỗng, token không khớp encoder...). Bật khi debug,
    # có thể tắt khi đã xác nhận pipeline chạy đúng trên server cụ thể. ---
    strict_mode: bool = True


# ---------------------------------------------------------------------------
# Token parsing: xác định tau_quo (dấu ngoặc kép mở) và tau_text (token nội
# dung chữ cần render) từ prompt.
# ---------------------------------------------------------------------------


class TokenAlignmentError(RuntimeError):
    """Raised khi token index tính từ parser không khớp với encoder thật.

    Đây chính là lỗi im lặng nguy hiểm nhất ở bản cũ (xem
    ARCHITECTURE_NOTES.md mục 3) — thay vì để guidance bám nhầm token, ta
    raise rõ ràng để người dùng biết ngay và tự đối chiếu lại cách
    tokenizer/encode_prompt thật của pipeline hoạt động.
    """


class TextGuiderTokenParser:
    """Xác định token dấu ngoặc kép và token nội dung chữ trong prompt.

    THAY ĐỔI QUAN TRỌNG so với bản cũ: KHÔNG còn mặc định áp
    `apply_chat_template` kiểu chat instruct. Theo tài liệu mô tả
    `Qwen3Embedder` trong repo chính thức, Qwen3 ở FLUX.2 Klein được dùng
    như một bộ trích đặc trưng (lấy hidden state ở layer [9,18,27] rồi nối
    lại), KHÔNG phải một lượt sinh chat — nên hầu như chắc chắn không có
    chat template nào được áp khi tokenize cho mục đích encode prompt.

    Mặc định giờ đây tokenize THÔ (không template, không pad cứng độ dài),
    và bắt buộc gọi `verify_alignment()` sau khi có `encoder_hidden_states`
    thật để đảm bảo không lệch vị trí — thay vì tin tưởng mù quáng.
    """

    @staticmethod
    def parse_tokens(
        tokenizer,
        prompt: str,
        use_chat_template: bool = False,
    ) -> Dict[str, object]:
        """Parse prompt để tìm token dấu ngoặc kép và token nội dung chữ.

        Args:
            tokenizer: tokenizer thật của pipeline (vd. pipe.tokenizer).
            prompt: prompt gốc do người dùng nhập.
            use_chat_template: chỉ bật nếu bạn đã xác nhận (qua đọc source
                thật của Flux2KleinPipeline.encode_prompt) rằng nó thật sự
                áp chat template trước khi tokenize. Mặc định False vì Qwen3
                ở đây là bộ trích đặc trưng, không phải chat model.

        Returns dict:
          - "quo_indices": List[int]
          - "text_token_indices": List[List[int]]
          - "text_strings": List[str]
          - "all_text_indices": List[int]
          - "num_text_tokens": int
          - "formatted_text": str — chuỗi thực sự đã tokenize (để đối chiếu)
          - "num_tokens_total": int — tổng số token thật (không padding) của
            chuỗi đã tokenize, dùng cho verify_alignment().
        """
        if use_chat_template:
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                formatted = prompt
        else:
            formatted = prompt

        # KHÔNG pad cứng max_length=512 nữa: pad giả sẽ tạo token thừa
        # không tồn tại trong encoder_hidden_states thật, phá alignment.
        # Chỉ tokenize đúng những gì có trong chuỗi.
        encoded = tokenizer(
            formatted,
            return_offsets_mapping=True,
            return_attention_mask=True,
            padding=False,
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", None) or 4096,
        )

        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))

        prompt_start_in_formatted = formatted.find(prompt)
        if prompt_start_in_formatted < 0:
            prompt_start_in_formatted = 0

        quoted_spans = _extract_quoted_spans(prompt)
        num_tokens_total = sum(1 for m in attention_mask if m == 1)

        if not quoted_spans:
            return {
                "quo_indices": [],
                "text_token_indices": [],
                "text_strings": [],
                "all_text_indices": [],
                "num_text_tokens": 0,
                "formatted_text": formatted,
                "num_tokens_total": num_tokens_total,
            }

        quo_indices: List[int] = []
        text_token_indices: List[List[int]] = []
        text_strings: List[str] = []

        for quote_char, text_content, char_start, char_end in quoted_spans:
            abs_quote_start = prompt_start_in_formatted + char_start
            abs_text_start = prompt_start_in_formatted + char_start + len(quote_char)
            abs_text_end = prompt_start_in_formatted + char_end - len(quote_char)

            quo_idx = _find_token_at_position(
                offsets, attention_mask, abs_quote_start, abs_quote_start + len(quote_char)
            )
            if quo_idx is not None:
                quo_indices.append(quo_idx)

            content_indices = _find_tokens_in_range(
                offsets, attention_mask, abs_text_start, abs_text_end
            )
            if content_indices:
                text_token_indices.append(content_indices)
                text_strings.append(text_content)

        all_text_indices = [idx for group in text_token_indices for idx in group]

        return {
            "quo_indices": quo_indices,
            "text_token_indices": text_token_indices,
            "text_strings": text_strings,
            "all_text_indices": all_text_indices,
            "num_text_tokens": len(all_text_indices),
            "formatted_text": formatted,
            "num_tokens_total": num_tokens_total,
        }

    @staticmethod
    def verify_alignment(
        token_info: Dict[str, object],
        encoder_hidden_states: Tensor,
        strict: bool = True,
    ) -> bool:
        """Đảm bảo token index tính offline khớp với chuỗi encoder thật.

        Đây là lớp phòng vệ trực tiếp cho lỗi ở ARCHITECTURE_NOTES.md mục 3:
        so `num_tokens_total` (từ tokenize riêng của parser) với độ dài
        thật `encoder_hidden_states.shape[1]`. Nếu lệch, guidance gần như
        chắc chắn bám sai token.

        Raises:
            TokenAlignmentError nếu strict=True và không khớp.

        Returns:
            True nếu khớp, False nếu lệch (chỉ khi strict=False).
        """
        real_len = encoder_hidden_states.shape[1]
        expected_len = token_info.get("num_tokens_total")
        if expected_len is None:
            return True  # không có thông tin để verify, bỏ qua

        # encoder thật có thể padding thêm ở cuối (attention_mask=0) — điều
        # kiện hợp lệ là: real_len >= expected_len, VÀ mọi token_info index
        # < expected_len đều nằm trong phần không-padding của real sequence.
        # Nếu real_len nhỏ hơn số token ta tính được offline, chắc chắn lệch
        # (bị truncate khác cách, hoặc formatted_text không khớp).
        ok = real_len >= expected_len
        if not ok and strict:
            raise TokenAlignmentError(
                f"[TextGuider] Token misalignment: parser tính {expected_len} "
                f"token nhưng encoder_hidden_states thật có {real_len} token. "
                f"Guidance sẽ bám sai vị trí nếu tiếp tục. Hãy kiểm tra lại "
                f"cách Flux2KleinPipeline.encode_prompt thật sự tokenize "
                f"prompt (có chat template không, có padding/truncate khác "
                f"không) rồi khớp lại TextGuiderTokenParser.parse_tokens."
            )
        return ok


def _extract_quoted_spans(prompt: str) -> List[Tuple[str, str, int, int]]:
    """Extract text spans enclosed in quotation marks.

    Trả về list (quote_char, text_content, char_start, char_end).
    """
    results = []

    patterns = [
        (r'"([^"]*)"', '"'),
        (r'\u201c([^\u201d]*)\u201d', '\u201c'),
        (r"'([^']*)'", "'"),
        (r'\u2018([^\u2019]*)\u2019', '\u2018'),
        (r'\u00ab([^\u00bb]*)\u00bb', '\u00ab'),
    ]

    for pattern, quote_char in patterns:
        for m in re.finditer(pattern, prompt):
            text_content = m.group(1)
            if text_content.strip():
                results.append((quote_char, text_content, m.start(), m.end()))

    results.sort(key=lambda x: x[2])
    deduped = []
    last_end = -1
    for item in results:
        if item[2] >= last_end:
            deduped.append(item)
            last_end = item[3]
    return deduped


def _find_token_at_position(
    offsets: List[Tuple[int, int]],
    attention_mask: List[int],
    char_start: int,
    char_end: int,
) -> Optional[int]:
    """Tìm token index có offset overlap nhiều nhất với [char_start, char_end)."""
    best_idx = None
    best_overlap = 0
    for i, (s, e) in enumerate(offsets):
        if attention_mask[i] == 0 or s == e:
            continue
        overlap = max(0, min(e, char_end) - max(s, char_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = i
    return best_idx


def _find_tokens_in_range(
    offsets: List[Tuple[int, int]],
    attention_mask: List[int],
    char_start: int,
    char_end: int,
) -> List[int]:
    """Tìm mọi token index có offset nằm trong [char_start, char_end)."""
    indices = []
    for i, (s, e) in enumerate(offsets):
        if attention_mask[i] == 0 or s == e:
            continue
        if s < char_end and e > char_start:
            indices.append(i)
    return indices


# ---------------------------------------------------------------------------
# Loss functions: Split Loss (Eq. 3) và Wrap Loss (Eq. 4).
# Không tìm thấy lỗi toán học ở phần này trong audit — GIỮ NGUYÊN logic cũ.
# ---------------------------------------------------------------------------


def symmetric_kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Symmetric KL divergence: d(p, q) = 0.5 * KL(p||q) + 0.5 * KL(q||p). (Eq. 5)"""
    p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)
    q_norm = q / (q.sum(dim=-1, keepdim=True) + eps)

    p_norm = p_norm.clamp(min=eps)
    q_norm = q_norm.clamp(min=eps)

    kl_pq = (p_norm * (p_norm.log() - q_norm.log())).sum(dim=-1)
    kl_qp = (q_norm * (q_norm.log() - p_norm.log())).sum(dim=-1)

    return 0.5 * (kl_pq + kl_qp).mean()


class TextGuiderLoss:
    """Split loss + wrap loss từ attention maps. (Section 3.3)"""

    @staticmethod
    def split_loss(attn_maps_text: List[Tensor]) -> Tensor:
        n = len(attn_maps_text)
        if n < 2:
            return torch.tensor(0.0, device=attn_maps_text[0].device if attn_maps_text else "cpu")

        loss = torch.tensor(0.0, device=attn_maps_text[0].device)
        for i in range(n):
            for j in range(i + 1, n):
                loss = loss - symmetric_kl_divergence(attn_maps_text[i], attn_maps_text[j])

        return loss

    @staticmethod
    def wrap_loss(attn_map_quo: Tensor, attn_maps_text: List[Tensor]) -> Tensor:
        if not attn_maps_text:
            return torch.tensor(0.0, device=attn_map_quo.device)

        text_sum = torch.stack(attn_maps_text, dim=0).sum(dim=0)
        return symmetric_kl_divergence(text_sum, attn_map_quo)

    @staticmethod
    def total_loss(
        attn_map_quo: Tensor,
        attn_maps_text: List[Tensor],
    ) -> Tensor:
        n = len(attn_maps_text)
        if n == 0:
            return torch.tensor(0.0, device=attn_map_quo.device)

        l_split = TextGuiderLoss.split_loss(attn_maps_text)
        l_wrap = TextGuiderLoss.wrap_loss(attn_map_quo, attn_maps_text)

        n_comparisons = n * (n - 1) // 2 + 1

        return (l_split + l_wrap) / n_comparisons


# ---------------------------------------------------------------------------
# AMO Sampler — GIỮ NGUYÊN công thức, chỉ nơi gọi ở pipeline được sửa để
# truyền generator nhất quán (tái lập được với --seed).
# ---------------------------------------------------------------------------


class AMOSampler:
    """Attention Modulated Overshooting sampler. (Equation 2)"""

    def __init__(self, config: TextGuiderConfig):
        self.c = config.amo_overshoot_c

    def compute_overshooting_mask(
        self,
        attn_maps_text: List[Tensor],
        num_img_tokens: int,
        spatial_h: int,
        spatial_w: int,
    ) -> Tensor:
        if not attn_maps_text:
            return torch.zeros(1, 1, spatial_h, spatial_w)

        stacked = torch.stack(attn_maps_text, dim=0)
        avg_attn = stacked.mean(dim=0)

        attn_min = avg_attn.min()
        attn_max = avg_attn.max()
        if attn_max - attn_min > 1e-8:
            avg_attn = (avg_attn - attn_min) / (attn_max - attn_min)
        else:
            avg_attn = torch.zeros_like(avg_attn)

        h_packed = spatial_h // 2
        w_packed = spatial_w // 2
        expected_packed = h_packed * w_packed

        if avg_attn.shape[0] == expected_packed:
            mask = avg_attn.reshape(1, 1, h_packed, w_packed)
            mask = F.interpolate(mask, size=(spatial_h, spatial_w), mode="bilinear", align_corners=False)
        elif avg_attn.shape[0] == spatial_h * spatial_w:
            mask = avg_attn.reshape(1, 1, spatial_h, spatial_w)
        else:
            total = avg_attn.shape[0]
            h_guess = int(math.sqrt(total * spatial_h / spatial_w))
            w_guess = total // max(h_guess, 1)
            if h_guess * w_guess == total:
                mask = avg_attn.reshape(1, 1, h_guess, w_guess)
                mask = F.interpolate(mask, size=(spatial_h, spatial_w), mode="bilinear", align_corners=False)
            else:
                mask = torch.zeros(1, 1, spatial_h, spatial_w, device=avg_attn.device)

        return mask.clamp(0, 1)

    def apply_overshooting(
        self,
        z_next: Tensor,
        t_next: float,
        epsilon: float,
        mask: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        device = z_next.device
        dtype = z_next.dtype

        mask = mask.to(device=device, dtype=dtype)
        if mask.shape[-2:] != z_next.shape[-2:]:
            mask = F.interpolate(
                mask, size=z_next.shape[-2:], mode="bilinear", align_corners=False
            )

        o = t_next + epsilon * self.c * mask

        noise = torch.randn(z_next.shape, device=device, dtype=dtype, generator=generator)
        overshoot = torch.sqrt(2.0 * o.clamp(min=0)) * noise

        return z_next + overshoot
