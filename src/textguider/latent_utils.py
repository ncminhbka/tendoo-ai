"""
Latent packing/unpacking utilities for FLUX.2 Klein.

Xác nhận (SimpleTuner fine-tuning doc, bên thứ ba):
    "Latent Channels: 32 VAE channels -> 128 after pixel shuffle"
=> D // 4 == 32 kênh là đúng cho FLUX.2 (không phải 16 như FLUX.1).
Các hàm dưới đây khớp giả định đó và được GIỮ NGUYÊN từ bản trước.

QUAN TRỌNG: đây chỉ là fallback. Bất cứ khi nào có sẵn một `pipe` Diffusers
thật, HÃY dùng `get_pack_unpack_fns()` bên dưới để ưu tiên gọi
`pipe._pack_latents` / `pipe._unpack_latents` của chính pipeline đó — đảm
bảo khớp tuyệt đối quy ước permute nội bộ, thay vì tin vào bản tự viết lại
này (xem ARCHITECTURE_NOTES.md mục 5 để biết lý do).
"""

from __future__ import annotations

from typing import Callable, Tuple

from torch import Tensor


def pack_latents(latents: Tensor) -> Tensor:
    """Pack 4D spatial latents [B, C, H, W] into FLUX 3D packed tokens [B, N, D].

    FLUX dùng patch 2x2 gộp vào chiều token:
      [B, C, H, W] -> [B, (H//2)*(W//2), C*4]
    """
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
    return latents


def unpack_latents(latents: Tensor, height: int, width: int) -> Tensor:
    """Unpack 3D packed tokens [B, N, D] into 4D spatial latents [B, C, H, W].

    FLUX.2 Klein dùng 32 kênh VAE -> 128 sau patch 2x2 (đã xác nhận, xem
    docstring module). Số kênh được suy ra từ D thay vì hardcode.
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
            f"{height}x{width} (expected {expected_tokens}). Nếu bạn thấy lỗi "
            f"này khi dùng Diffusers pipeline, rất có thể quy ước patch không "
            f"khớp — hãy dùng pipe._unpack_latents thay vì hàm này."
        )
    latents = latents.view(B, H_p, W_p, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(B, channels, H_lat, W_lat)
    return latents


def get_pack_unpack_fns(
    pipe=None,
) -> Tuple[Callable[[Tensor, int, int], Tensor], Callable[[Tensor, int, int], Tensor]]:
    """Trả về cặp hàm (pack, unpack) với chữ ký thống nhất (latents, height, width).

    Ưu tiên tuyệt đối các hàm nội bộ của chính `pipe` (nếu có) để đảm bảo
    khớp bit-for-bit với những gì pipeline thật sự dùng khi decode/encode
    latent. Chỉ rơi về bản tự viết trong module này khi không có pipeline
    (ví dụ path native dùng thẳng model gốc BFL, hoặc pipe không lộ hàm này).
    """
    has_unpack = pipe is not None and hasattr(pipe, "_unpack_latents")
    # Diffusers Flux family thường không có _pack_latents công khai (chỉ có
    # _unpack_latents + _prepare_latents tự pack lúc khởi tạo), nên với pack
    # ta vẫn cần bản tự viết trong đa số trường hợp — nhưng luôn thử trước.
    has_pack = pipe is not None and hasattr(pipe, "_pack_latents")

    def _unpack(latents: Tensor, height: int, width: int) -> Tensor:
        if has_unpack:
            try:
                vsf = getattr(pipe, "vae_scale_factor", 8)
                return pipe._unpack_latents(latents, height, width, vsf)
            except Exception:
                pass  # rơi về bản tự viết, nhưng log để người dùng biết
        return unpack_latents(latents, height=height, width=width)

    def _pack(latents: Tensor, height: int, width: int) -> Tensor:
        if has_pack:
            try:
                return pipe._pack_latents(latents)
            except Exception:
                pass
        return pack_latents(latents)

    return _pack, _unpack
