"""
TextGuider: Training-Free Guidance for Text Rendering via Attention Alignment.
arXiv:2512.09350

Core implementation of TextGuider loss functions (split loss, wrap loss),
AMO Sampler integration, and token parsing for FLUX.2 Klein 4B base.

Key components:
  - TextGuiderConfig: hyperparameters for guidance
  - TextGuiderTokenParser: identifies quotation mark and textual content tokens
  - TextGuiderLoss: split loss + wrap loss computation
  - AMOSampler: Attention Modulated Overshooting mechanism
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TextGuiderConfig:
    """Configuration hyperparameters for TextGuider.

    Paper defaults (Section 4.1, Section B, Section D):
      - alpha = 60 (guidance step size)
      - t_guide_ratio = 0.25 (first quarter of denoising steps)
      - amo_overshoot_c = 0.5
      - 100 denoising steps, 512x512 resolution in the paper
      - For FLUX.2 Klein base (50 steps default), ratios are preserved.
    """

    # --- TextGuider guidance ---
    alpha: float = 60.0  # Guidance step size (Eq. 6 update rule)
    t_guide_ratio: float = 0.25  # Fraction of total steps to apply guidance

    # --- AMO Sampler (Equation 2) ---
    amo_enabled: bool = True  # Enable Attention Modulated Overshooting
    amo_overshoot_c: float = 0.5  # Overshooting hyperparameter c

    # --- Gradient computation ---
    target_layers: str = "double"  # "double" = dual-stream blocks only (paper Sec. B)
    use_gradient_checkpointing: bool = True  # Memory-efficient backprop

    # --- Generation defaults for FLUX.2 Klein base ---
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    resolution: int = 1024


# ---------------------------------------------------------------------------
# Token parsing: identify τ_quo (opening quotation mark) and τ_text (textual
# content tokens) from the tokenized prompt.
# ---------------------------------------------------------------------------

# Unicode quotation mark characters to search for in the token sequence.
QUOTATION_MARKS = {'"', '\u201c', '\u201d', '\u2018', '\u2019', "'", '\u00ab', '\u00bb'}
# Opening quotation marks specifically.
OPENING_QUOTES = {'"', '\u201c', '\u2018', "'", '\u00ab'}


class TextGuiderTokenParser:
    """Identifies quotation-mark tokens and textual-content tokens in a prompt.

    In the TextGuider paper, prompts use quotation marks to delimit the text
    that should be rendered in the image.  For example:
        'A sign that says "Hello World"'
    Here τ_quo = token index of the opening `"`, and τ_text = token indices
    for "Hello" and "World".

    This parser handles Qwen3's chat-template formatting used by FLUX.2 Klein.
    """

    @staticmethod
    def parse_tokens(
        tokenizer,
        prompt: str,
    ) -> Dict[str, object]:
        """Parse a prompt to find quotation marks and textual content tokens.

        Returns a dict with:
          - "quo_indices": List[int]  — token positions of opening quotation marks
          - "text_token_indices": List[List[int]] — per-quoted-span token positions
          - "text_strings": List[str] — the raw text strings within quotes
          - "all_text_indices": List[int] — flattened list of all text token indices
          - "num_text_tokens": int — total number of textual content tokens
        """
        # 1. Apply Qwen3 chat template (same as FLUX.2 Klein pipeline)
        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            formatted = prompt

        # 2. Tokenize with offset mapping
        encoded = tokenizer(
            formatted,
            return_offsets_mapping=True,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))

        # 3. Find the prompt text within the formatted template
        prompt_start_in_formatted = formatted.find(prompt)
        if prompt_start_in_formatted < 0:
            prompt_start_in_formatted = 0
        prompt_end_in_formatted = prompt_start_in_formatted + len(prompt)

        # 4. Extract quoted text spans from the original prompt
        quoted_spans = _extract_quoted_spans(prompt)
        if not quoted_spans:
            return {
                "quo_indices": [],
                "text_token_indices": [],
                "text_strings": [],
                "all_text_indices": [],
                "num_text_tokens": 0,
            }

        # 5. For each quoted span, find the token indices
        quo_indices: List[int] = []
        text_token_indices: List[List[int]] = []
        text_strings: List[str] = []

        for quote_char, text_content, char_start, char_end in quoted_spans:
            # Adjust positions to the formatted template
            abs_quote_start = prompt_start_in_formatted + char_start
            abs_text_start = prompt_start_in_formatted + char_start + len(quote_char)
            abs_text_end = prompt_start_in_formatted + char_end - len(quote_char)

            # Find the opening quotation mark token
            quo_idx = _find_token_at_position(
                offsets, attention_mask, abs_quote_start, abs_quote_start + len(quote_char)
            )
            if quo_idx is not None:
                quo_indices.append(quo_idx)

            # Find all textual content tokens
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
        }


def _extract_quoted_spans(prompt: str) -> List[Tuple[str, str, int, int]]:
    """Extract text spans enclosed in quotation marks.

    Returns list of (quote_char, text_content, char_start, char_end).
    char_start/end are positions in the original prompt string,
    where char_start points to the opening quote and char_end points
    past the closing quote.
    """
    results = []

    # Match various quotation patterns
    patterns = [
        (r'"([^"]*)"', '"'),       # Standard double quotes
        (r'\u201c([^\u201d]*)\u201d', '\u201c'),  # "curly" quotes
        (r"'([^']*)'", "'"),       # Single quotes (use carefully)
        (r'\u2018([^\u2019]*)\u2019', '\u2018'),  # 'curly' single quotes
        (r'\u00ab([^\u00bb]*)\u00bb', '\u00ab'),  # «guillemets»
    ]

    for pattern, quote_char in patterns:
        for m in re.finditer(pattern, prompt):
            text_content = m.group(1)
            if text_content.strip():  # Skip empty quotes
                results.append((quote_char, text_content, m.start(), m.end()))

    # Sort by position and deduplicate overlapping spans
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
    """Find the token index whose offset overlaps [char_start, char_end)."""
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
    """Find all token indices whose offsets fall within [char_start, char_end)."""
    indices = []
    for i, (s, e) in enumerate(offsets):
        if attention_mask[i] == 0 or s == e:
            continue
        # Token overlaps with the target range
        if s < char_end and e > char_start:
            indices.append(i)
    return indices


# ---------------------------------------------------------------------------
# Loss functions: Split Loss (Eq. 3) and Wrap Loss (Eq. 4)
# ---------------------------------------------------------------------------

def symmetric_kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Symmetric KL divergence: d(p, q) = 0.5 * KL(p||q) + 0.5 * KL(q||p).

    Paper Equation 5.  Inputs are normalized to sum to 1 to ensure valid
    probability distributions.

    Args:
        p: Attention map, shape [..., N] (last dim is spatial)
        q: Attention map, shape [..., N]
        eps: Small constant for numerical stability

    Returns:
        Scalar symmetric KL divergence.
    """
    # Normalize to probability distributions
    p_norm = p / (p.sum(dim=-1, keepdim=True) + eps)
    q_norm = q / (q.sum(dim=-1, keepdim=True) + eps)

    # Clamp for log stability
    p_norm = p_norm.clamp(min=eps)
    q_norm = q_norm.clamp(min=eps)

    kl_pq = (p_norm * (p_norm.log() - q_norm.log())).sum(dim=-1)
    kl_qp = (q_norm * (q_norm.log() - p_norm.log())).sum(dim=-1)

    return 0.5 * (kl_pq + kl_qp).mean()


class TextGuiderLoss:
    """Computes TextGuider's split loss and wrap loss from attention maps.

    Paper Section 3.3:
      - Split loss (Eq. 3): Encourages spatially separated activations for
        each textual content token.
      - Wrap loss (Eq. 4): Encourages the quotation mark token attention to
        cover all textual content token regions.
      - Total loss (Eq. 6): L = (L_split + L_wrap) / N
        where N = C(n,2) + 1, normalizing by the number of comparisons.
    """

    @staticmethod
    def split_loss(attn_maps_text: List[Tensor]) -> Tensor:
        """Equation 3: Split loss over all pairs of textual content tokens.

        Minimizes overlap between attention maps of different text tokens,
        encouraging each token to activate in its own spatial region.

        Args:
            attn_maps_text: List of attention maps for each textual content
                token, each shape [num_img_tokens] (averaged over heads/layers).

        Returns:
            Scalar split loss.
        """
        n = len(attn_maps_text)
        if n < 2:
            return torch.tensor(0.0, device=attn_maps_text[0].device if attn_maps_text else "cpu")

        loss = torch.tensor(0.0, device=attn_maps_text[0].device)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Negative symmetric KL: we want to MAXIMIZE divergence
                # (minimize overlap), so we minimize -d(A_τi, A_τj)
                loss = loss - symmetric_kl_divergence(attn_maps_text[i], attn_maps_text[j])
                count += 1

        return loss  # Already negative: minimizing this maximizes separation

    @staticmethod
    def wrap_loss(attn_map_quo: Tensor, attn_maps_text: List[Tensor]) -> Tensor:
        """Equation 4: Wrap loss ensuring quotation mark covers all text token regions.

        L_wrap = D_SKL( norm(sum_{i=1}^n A_{tau_i}), norm(A_{tau_quo}) )

        Minimizes the symmetric KL divergence between the quotation mark
        attention and the sum of all textual content token attentions,
        encouraging tau_quo to attend broadly over the entire text region.

        Args:
            attn_map_quo: Attention map for the opening quotation mark token,
                shape [num_img_tokens].
            attn_maps_text: List of attention maps for textual content tokens.

        Returns:
            Scalar wrap loss.
        """
        if not attn_maps_text:
            return torch.tensor(0.0, device=attn_map_quo.device)

        # Sum of textual content attentions (Eq. 4: sum_{i=1}^n A_{tau_i})
        text_sum = torch.stack(attn_maps_text, dim=0).sum(dim=0)

        # Minimize symmetric KL between sum of text token attentions and quotation mark attention
        return symmetric_kl_divergence(text_sum, attn_map_quo)

    @staticmethod
    def total_loss(
        attn_map_quo: Tensor,
        attn_maps_text: List[Tensor],
    ) -> Tensor:
        """Equation 6: Combined normalized loss.

        L = (L_split + L_wrap) / N
        where N = C(n, 2) + 1 accounts for the number of pairwise comparisons.

        Args:
            attn_map_quo: Attention map for opening quotation mark token.
            attn_maps_text: List of attention maps for textual content tokens.

        Returns:
            Scalar total loss.
        """
        n = len(attn_maps_text)
        if n == 0:
            return torch.tensor(0.0, device=attn_map_quo.device)

        l_split = TextGuiderLoss.split_loss(attn_maps_text)
        l_wrap = TextGuiderLoss.wrap_loss(attn_map_quo, attn_maps_text)

        # N = C(n, 2) + 1 = n*(n-1)/2 + 1
        n_comparisons = n * (n - 1) // 2 + 1

        return (l_split + l_wrap) / n_comparisons


# ---------------------------------------------------------------------------
# AMO Sampler: Attention Modulated Overshooting (from AMO Sampler paper,
# integrated as described in TextGuider Section 3.3)
# ---------------------------------------------------------------------------

class AMOSampler:
    """Attention Modulated Overshooting sampler.

    Paper Equation 2:
      Z_{t_{k+1}} = Z_{t_k} + ε * v_θ(Z_{t_k}, t_k)
                     + sqrt(2 * o) * ξ ⊙ o
    where:
      o = t_{k+1} + ε * c * m
      ξ ~ N(0, I)
      m = attention mask from cross-modal attention
      c = overshooting hyperparameter

    TextGuider uses A (image-query, text-key) for both guidance and
    AMO mask, instead of A^rev (text-query, image-key) used by original AMO.
    """

    def __init__(self, config: TextGuiderConfig):
        self.c = config.amo_overshoot_c

    def compute_overshooting_mask(
        self,
        attn_maps_text: List[Tensor],
        num_img_tokens: int,
        spatial_h: int,
        spatial_w: int,
    ) -> Tensor:
        """Compute the attention-derived spatial mask m for AMO overshooting.

        The mask is the average of cross-modal attention maps across all
        textual content tokens, reshaped to the spatial latent dimensions.

        Args:
            attn_maps_text: Per-token attention maps, each [num_img_tokens].
            num_img_tokens: Expected number of image tokens.
            spatial_h: Latent height.
            spatial_w: Latent width.

        Returns:
            Mask tensor of shape [1, 1, spatial_h, spatial_w], values in [0, 1].
        """
        if not attn_maps_text:
            return torch.zeros(1, 1, spatial_h, spatial_w)

        # Average attention across all text tokens
        stacked = torch.stack(attn_maps_text, dim=0)  # [n, num_img_tokens]
        avg_attn = stacked.mean(dim=0)  # [num_img_tokens]

        # Normalize to [0, 1]
        attn_min = avg_attn.min()
        attn_max = avg_attn.max()
        if attn_max - attn_min > 1e-8:
            avg_attn = (avg_attn - attn_min) / (attn_max - attn_min)
        else:
            avg_attn = torch.zeros_like(avg_attn)

        # Reshape to spatial dimensions
        # FLUX.2 Klein uses 2x2 packing, so num_img_tokens = (H/16) * (W/16)
        # The attention map over image tokens maps to spatial_h/2 * spatial_w/2
        h_packed = spatial_h // 2
        w_packed = spatial_w // 2
        expected_packed = h_packed * w_packed

        if avg_attn.shape[0] == expected_packed:
            mask = avg_attn.reshape(1, 1, h_packed, w_packed)
            # Upsample to full latent resolution
            mask = F.interpolate(mask, size=(spatial_h, spatial_w), mode="bilinear", align_corners=False)
        elif avg_attn.shape[0] == spatial_h * spatial_w:
            mask = avg_attn.reshape(1, 1, spatial_h, spatial_w)
        else:
            # Fallback: try to find closest spatial arrangement
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
        """Apply AMO overshooting step (Equation 2, second term).

        Args:
            z_next: Latent after standard Euler step, shape [B, C, H, W].
            t_next: Next timestep value t_{k+1}.
            epsilon: Step size ε = t_{k+1} - t_k.
            mask: Spatial attention mask m, shape [1, 1, H, W].
            generator: Optional random generator for reproducibility.

        Returns:
            Updated latent with overshooting noise applied.
        """
        device = z_next.device
        dtype = z_next.dtype

        mask = mask.to(device=device, dtype=dtype)
        if mask.shape[-2:] != z_next.shape[-2:]:
            mask = F.interpolate(
                mask, size=z_next.shape[-2:], mode="bilinear", align_corners=False
            )

        # o = t_{k+1} + ε * c * m
        o = t_next + epsilon * self.c * mask

        # sqrt(2 * o) * ξ where ξ ~ N(0, I)
        noise = torch.randn(z_next.shape, device=device, dtype=dtype, generator=generator)
        overshoot = torch.sqrt(2.0 * o.clamp(min=0)) * noise

        return z_next + overshoot
