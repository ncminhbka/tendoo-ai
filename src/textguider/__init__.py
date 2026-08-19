"""
TextGuider: Training-Free Guidance for Text Rendering via Attention Alignment.
arXiv:2512.09350

Separate implementation from FreeText (arXiv:2601.00535).
Provides training-free latent guidance for FLUX.2 Klein 4B base to improve
text rendering by aligning quotation mark and textual content token attention.
"""

from .textguider import (
    TextGuiderConfig,
    TextGuiderLoss,
    TextGuiderTokenParser,
    AMOSampler,
    symmetric_kl_divergence,
)
from .textguider_attention import TextGuiderAttentionStore, TextGuiderForwardWrapper
from .pipeline_textguider import TextGuiderFluxPipeline

__all__ = [
    "TextGuiderConfig",
    "TextGuiderLoss",
    "TextGuiderTokenParser",
    "AMOSampler",
    "symmetric_kl_divergence",
    "TextGuiderAttentionStore",
    "TextGuiderForwardWrapper",
    "TextGuiderFluxPipeline",
]
