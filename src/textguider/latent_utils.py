"""
Latent packing/unpacking utilities for FLUX.2 Klein.

Standalone copy for the TextGuider package, avoiding dependency on FreeText.
These functions convert between FLUX's 3D packed token format [B, N, D]
and standard 4D spatial latents [B, C, H, W].
"""

from __future__ import annotations

import torch
from torch import Tensor


def pack_latents(latents: Tensor) -> Tensor:
    """Pack 4D spatial latents [B, C, H, W] into FLUX 3D packed tokens [B, N, D].

    FLUX uses 2x2 spatial patches packed into the token dimension:
      [B, C, H, W] → [B, (H//2)*(W//2), C*4]
    """
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
    return latents


def unpack_latents(latents: Tensor, height: int, width: int) -> Tensor:
    """Unpack 3D packed tokens [B, N, D] into 4D spatial latents [B, C, H, W].

    FLUX.2 Klein 4B uses [B, N, 128] / 32 channels. The channel count is
    inferred from the token width.
    """
    B, N, D = latents.shape
    if D % 4 != 0:
        raise ValueError(f"Packed latent width must be divisible by 4, got {D}")
    channels = D // 4
    H_lat = height // 8
    W_lat = width // 8
    H_p = H_lat // 2
    W_p = W_lat // 2
    expected_tokens = H_p * W_p
    if N != expected_tokens:
        raise ValueError(
            f"Packed latent token count {N} does not match resolution "
            f"{height}x{width} (expected {expected_tokens})"
        )
    latents = latents.view(B, H_p, W_p, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(B, channels, H_lat, W_lat)
    return latents
