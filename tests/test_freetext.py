"""
Unit and Dry-Run Tests for FreeText Modules.
Verifies Vietnamese Glyph Rendering, 2D Log-Gabor FFT Filter, Noise Alignment, and SGMI.
"""

import unittest
import os
import sys
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from freetext.glyph_renderer import GlyphRenderer, extract_text_spans
from freetext.log_gabor import LogGaborFilter
from freetext.localization import AttentionLocalization, otsu_threshold
from freetext.sgmi import FreeTextConfig, SpectralGlyphInjector
from freetext.pipeline_flux_freetext import FreeTextFluxPipeline


class TestFreeText(unittest.TestCase):

    def test_extract_text_spans(self):
        prompt = 'Banner khuyến mại "PHỞ BÒ GIA TRUYỀN", giảm giá \'ƯU ĐÃI 50%\' và chữ “KHAI TRƯƠNG”.'
        spans = extract_text_spans(prompt)
        self.assertIn("PHỞ BÒ GIA TRUYỀN", spans)
        self.assertIn("ƯU ĐÃI 50%", spans)
        self.assertIn("KHAI TRƯƠNG", spans)

    def test_glyph_renderer(self):
        renderer = GlyphRenderer()
        texts = ["PHỞ BÒ", "GIẢM GIÁ 50%"]
        img_tensor, mask_tensor, regions = renderer.get_glyph_tensor(
            texts=texts,
            width=512,
            height=512,
            device="cpu",
            dtype=torch.float32,
        )
        self.assertEqual(img_tensor.shape, (1, 3, 512, 512))
        self.assertEqual(mask_tensor.shape, (1, 1, 512, 512))
        self.assertEqual(len(regions), 2)
        self.assertTrue(mask_tensor.max() > 0)

    def test_log_gabor_filter(self):
        filter_2d = LogGaborFilter(center_freq=0.22, bandwidth_ratio=0.55)
        # Test filter construction
        gabor_grid = filter_2d.construct_2d_filter(64, 64)
        self.assertEqual(gabor_grid.shape, (1, 1, 64, 64))
        # Center DC component should be 0
        self.assertEqual(float(gabor_grid[0, 0, 32, 32]), 0.0)

        # Test application on latent tensor [1, 16, 64, 64]
        latents = torch.randn((1, 16, 64, 64), dtype=torch.float32)
        modulated = filter_2d.apply_spectral_modulation(latents)
        self.assertEqual(modulated.shape, latents.shape)
        self.assertFalse(torch.isnan(modulated).any())

    def test_otsu_threshold(self):
        # Array with two clear distinct clusters
        arr = np.concatenate([np.ones((50, 50)) * 0.1, np.ones((50, 50)) * 0.9])
        thresh = otsu_threshold(arr)
        self.assertTrue(0.1 < thresh < 0.9)

    def test_sgmi_injector_lifecycle(self):
        config = FreeTextConfig(enabled=True, t_start=0.2, t_end=0.8, injection_strength=0.85)
        injector = SpectralGlyphInjector(config=config)

        # Weight annealing curve
        self.assertEqual(injector.compute_annealed_weight(0.1), 0.0)
        self.assertEqual(injector.compute_annealed_weight(0.9), 0.0)
        mid_weight = injector.compute_annealed_weight(0.5)
        self.assertTrue(mid_weight > 0.5)

        # Mock prepare with prompt
        prompt = 'Banner "QUÁN ĂN NGON"'
        has_text = injector.prepare(prompt=prompt, vae=None, height=256, width=256)
        self.assertTrue(has_text)
        self.assertIsNotNone(injector.z0_glyph)
        self.assertIsNotNone(injector.mask)

        # Mock step injection on 4D latents
        latents_4d = torch.randn_like(injector.z0_glyph)
        updated_4d = injector.inject_step(latents_4d, progress=0.5)
        self.assertEqual(updated_4d.shape, latents_4d.shape)

        # Mock step injection on FLUX 3D packed tokens [B, N, 64]
        H_lat, W_lat = injector.z0_glyph.shape[-2], injector.z0_glyph.shape[-1]
        num_patches = (H_lat // 2) * (W_lat // 2)
        latents_3d = torch.randn((1, num_patches, 64), dtype=torch.float32)
        updated_3d = injector.inject_step(latents_3d, progress=0.5)
        self.assertEqual(updated_3d.shape, (1, num_patches, 64))
        self.assertFalse(torch.isnan(updated_3d).any())

    def test_pipeline_dry_run(self):
        pipe = FreeTextFluxPipeline(pipe=None)
        img = pipe.generate(
            prompt='Poster quán cafe "CÀ PHÊ TRỨNG HÀ NỘI"',
            width=512,
            height=512,
            use_freetext=True,
        )
        self.assertEqual(img.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
