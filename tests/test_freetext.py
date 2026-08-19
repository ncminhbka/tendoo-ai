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
from freetext.sgmi import (
    FreeTextConfig,
    SpectralGlyphInjector,
    patchify_vae_latents,
    unpatchify_vae_latents,
)
from freetext.pipeline_flux_freetext import FreeTextFluxPipeline
from freetext.attention_capture import Flux2AttentionRecorder


class TestFreeText(unittest.TestCase):

    def test_extract_text_spans(self):
        prompt = 'Banner khuyến mại "PHỞ BÒ GIA TRUYỀN", giảm giá \'ƯU ĐÃI 50%\' và chữ “KHAI TRƯƠNG”.'
        spans = extract_text_spans(prompt)
        self.assertIn("PHỞ BÒ GIA TRUYỀN", spans)
        self.assertIn("ƯU ĐÃI 50%", spans)
        self.assertIn("KHAI TRƯƠNG", spans)

    def test_flux2_token_mapping_matches_chat_template_sequence(self):
        class FakeTokenizer:
            all_special_ids = [99]

            def apply_chat_template(self, messages, **kwargs):
                return "<chat>" + messages[0]["content"] + "</chat>"

            def __call__(self, text, **kwargs):
                max_length = kwargs.get("max_length", len(text))
                ids = list(range(len(text))) + [99] * max(0, max_length - len(text))
                offsets = [(i, i + 1) for i in range(len(text))] + [(0, 0)] * max(0, max_length - len(text))
                mask = [1] * len(text) + [0] * max(0, max_length - len(text))
                return {"input_ids": ids[:max_length], "offset_mapping": offsets[:max_length], "attention_mask": mask[:max_length]}

        groups, sinks = AttentionLocalization._token_groups(
            FakeTokenizer(), "PHỞ BÒ", ["BÒ"]
        )
        self.assertEqual(groups, [[10, 11]])
        self.assertEqual(sinks, [0, 11])

    def test_callback_uses_post_scheduler_sigma(self):
        wrapper = FreeTextFluxPipeline(pipe=None, device="cpu", dtype=torch.float32)
        seen = {}

        def record(latents, progress, timestep_sigma=None):
            seen["progress"] = progress
            seen["sigma"] = timestep_sigma
            return latents

        wrapper.injector.inject_step = record

        class FakeScheduler:
            sigmas = torch.tensor([1.0, 0.75, 0.5, 0.0])

        class FakePipe:
            scheduler = FakeScheduler()

        callback = wrapper.create_step_callback(num_inference_steps=3)
        callback(FakePipe(), 0, torch.tensor(1.0), {"latents": torch.zeros(1)})
        self.assertAlmostEqual(seen["sigma"], 0.75)
        self.assertAlmostEqual(seen["progress"], 0.25)

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

    def test_flux2_vae_patch_roundtrip(self):
        latents = torch.randn((1, 32, 16, 16), dtype=torch.float32)
        patched = patchify_vae_latents(latents)
        self.assertEqual(patched.shape, (1, 128, 8, 8))
        restored = unpatchify_vae_latents(patched)
        self.assertTrue(torch.equal(latents, restored))

    def test_otsu_threshold(self):
        # Array with two clear distinct clusters
        arr = np.concatenate([np.ones((50, 50)) * 0.1, np.ones((50, 50)) * 0.9])
        thresh = otsu_threshold(arr)
        self.assertTrue(0.1 < thresh < 0.9)

    def test_attention_recorder_uses_joint_denominator(self):
        recorder = Flux2AttentionRecorder(target_groups=[[0]], sink_indices=[1], max_query_chunk=2)
        query = torch.randn((1, 4, 2, 3), dtype=torch.float32)
        keys = torch.randn((1, 6, 2, 3), dtype=torch.float32)
        recorder.capture("block.0", query, keys, text_len=2)
        recorder.finalize_step(3)
        self.assertEqual(len(recorder.records), 1)
        self.assertEqual(recorder.records[0]["map"].shape, (1, 4))
        self.assertTrue(torch.isfinite(recorder.records[0]["map"]).all())

    def test_attention_recorder_keeps_conditional_map_under_cfg(self):
        recorder = Flux2AttentionRecorder(target_groups=[[0]], sink_indices=[], max_query_chunk=2)
        conditional_query = torch.ones((1, 4, 2, 3), dtype=torch.float32)
        unconditional_query = torch.full((1, 4, 2, 3), -1.0, dtype=torch.float32)
        keys = torch.randn((1, 2, 2, 3), dtype=torch.float32)
        recorder.capture("block.0", conditional_query, keys, text_len=1)
        recorder.capture("block.0", unconditional_query, keys, text_len=1)
        recorder.finalize_step(0)
        self.assertEqual(len(recorder.records), 1)
        expected = Flux2AttentionRecorder(target_groups=[[0]], sink_indices=[], max_query_chunk=2)
        expected.capture("block.0", conditional_query, keys, text_len=1)
        expected.finalize_step(0)
        self.assertTrue(torch.equal(recorder.records[0]["map"], expected.records[0]["map"]))

    def test_attention_localization_selects_refined_mask(self):
        localization = AttentionLocalization(top_k_pairs=2)
        base = torch.zeros((1, 16), dtype=torch.float32)
        base[:, 5:7] = 1.0
        localization.attention_records = [
            {"step": 5, "layer": "block.0", "target": 0, "map": base},
            {"step": 6, "layer": "block.1", "target": 0, "map": base * 0.9},
        ]
        mask = localization.build_attention_mask(
            regions=[{"box": (4, 0, 12, 16)}],
            target_h=8,
            target_w=8,
            img_w=16,
            img_h=16,
        )
        self.assertEqual(mask.shape, (1, 1, 8, 8))
        self.assertGreater(float(mask.sum()), 0.0)

    def test_sgmi_injector_lifecycle(self):
        config = FreeTextConfig(enabled=True, t_start=0.2, t_end=0.4, injection_strength=1.0)
        injector = SpectralGlyphInjector(config=config)

        # Weight annealing curve
        self.assertEqual(injector.compute_annealed_weight(0.1), 0.0)
        self.assertEqual(injector.compute_annealed_weight(0.5), 0.0)
        start_weight = injector.compute_annealed_weight(0.2)
        self.assertAlmostEqual(start_weight, 1.0, places=5)

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

        # FLUX.2 Klein uses 32 spatial channels packed into width 128 tokens.
        injector.z0_glyph = torch.randn((1, 32, H_lat, W_lat), dtype=torch.float32)
        injector.noise_ref = torch.randn_like(injector.z0_glyph)
        latents_klein = torch.randn((1, num_patches, 128), dtype=torch.float32)
        updated_klein = injector.inject_step(latents_klein, progress=0.5)
        self.assertEqual(updated_klein.shape, (1, num_patches, 128))
        self.assertFalse(torch.isnan(updated_klein).any())

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
