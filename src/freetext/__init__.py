"""
FreeText: Training-Free Text Rendering in Diffusion Transformers
for Vietnamese Banner and Poster Generation.
arXiv:2601.00535
"""

from .glyph_renderer import GlyphRenderer, extract_text_spans, get_vietnamese_font
from .log_gabor import LogGaborFilter
from .localization import AttentionLocalization, otsu_threshold
from .sgmi import FreeTextConfig, SpectralGlyphInjector
from .pipeline_flux_freetext import FreeTextFluxPipeline

__all__ = [
    "GlyphRenderer",
    "extract_text_spans",
    "get_vietnamese_font",
    "LogGaborFilter",
    "AttentionLocalization",
    "otsu_threshold",
    "FreeTextConfig",
    "SpectralGlyphInjector",
    "FreeTextFluxPipeline",
]
