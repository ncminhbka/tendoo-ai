"""
TextGuider: Training-Free Guidance for Text Rendering via Attention Alignment.
arXiv:2512.09350

Separate implementation from FreeText (arXiv:2601.00535).
Provides training-free latent guidance for FLUX.2 Klein 4B base to improve
text rendering by aligning quotation mark and textual content token attention.

Xem ARCHITECTURE_NOTES.md trong cùng thư mục để biết chi tiết các sự thật
kiến trúc FLUX.2 đã xác nhận từ nguồn chính thức và những gì đã được sửa
so với bản trước.
"""

from .textguider import (
    TextGuiderConfig,
    TextGuiderLoss,
    TextGuiderTokenParser,
    TokenAlignmentError,
    AMOSampler,
    symmetric_kl_divergence,
)
from .textguider_attention import (
    TextGuiderAttentionStore,
    TextGuiderForwardWrapper,
    AttentionCaptureError,
    build_guidance_tensor,
)
from .pipeline_textguider import TextGuiderFluxPipeline

__all__ = [
    "TextGuiderConfig",
    "TextGuiderLoss",
    "TextGuiderTokenParser",
    "TokenAlignmentError",
    "AMOSampler",
    "symmetric_kl_divergence",
    "TextGuiderAttentionStore",
    "TextGuiderForwardWrapper",
    "AttentionCaptureError",
    "build_guidance_tensor",
    "TextGuiderFluxPipeline",
]
