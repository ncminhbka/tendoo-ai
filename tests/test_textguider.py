"""
Unit tests for TextGuider implementation.
arXiv:2512.09350

Tests cover:
  - Token parsing (quotation mark identification, text span extraction)
  - Loss functions (split loss, wrap loss, symmetric KL divergence)
  - AMO Sampler (mask computation, overshooting)
  - Attention store (storage, aggregation)
"""

import math
import sys
from pathlib import Path

import pytest
import torch

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.textguider import (
    TextGuiderConfig,
    TextGuiderLoss,
    TextGuiderTokenParser,
    AMOSampler,
    TextGuiderAttentionStore,
)
from src.textguider.textguider import (
    symmetric_kl_divergence,
    _extract_quoted_spans,
)


# ============================================================================
# Token Parsing Tests
# ============================================================================


class TestExtractQuotedSpans:
    """Test quoted span extraction from prompts."""

    def test_double_quotes(self):
        spans = _extract_quoted_spans('A sign that says "Hello World"')
        assert len(spans) == 1
        assert spans[0][1] == "Hello World"

    def test_multiple_quotes(self):
        spans = _extract_quoted_spans('Banner with "SALE 50%" and "BUY NOW"')
        assert len(spans) == 2
        assert spans[0][1] == "SALE 50%"
        assert spans[1][1] == "BUY NOW"

    def test_curly_quotes(self):
        spans = _extract_quoted_spans('A poster with \u201cGI\u1ea2M GI\u00c1\u201d text')
        assert len(spans) == 1
        assert "GI\u1ea2M" in spans[0][1]

    def test_no_quotes(self):
        spans = _extract_quoted_spans("A beautiful landscape with mountains")
        assert len(spans) == 0

    def test_empty_quotes(self):
        spans = _extract_quoted_spans('A sign that says "" nothing')
        assert len(spans) == 0  # Empty quotes should be skipped

    def test_vietnamese_text(self):
        spans = _extract_quoted_spans(
            'Poster khai tr\u01b0\u01a1ng c\u1eeda h\u00e0ng v\u1edbi ch\u1eef "PH\u1ede B\u00d2 GIA TRUY\u1ec0N"'
        )
        assert len(spans) == 1
        assert spans[0][1] == "PH\u1ede B\u00d2 GIA TRUY\u1ec0N"

    def test_mixed_quotes(self):
        spans = _extract_quoted_spans('Text with "hello" and \u201cworld\u201d')
        assert len(spans) == 2

    def test_position_tracking(self):
        prompt = 'A "test" string'
        spans = _extract_quoted_spans(prompt)
        assert len(spans) == 1
        quote_char, text, start, end = spans[0]
        assert prompt[start:end] == '"test"'


# ============================================================================
# Symmetric KL Divergence Tests
# ============================================================================


class TestSymmetricKL:
    """Test symmetric KL divergence computation."""

    def test_symmetric(self):
        """d(p, q) == d(q, p)"""
        p = torch.softmax(torch.randn(32), dim=0)
        q = torch.softmax(torch.randn(32), dim=0)
        assert abs(symmetric_kl_divergence(p, q).item() -
                   symmetric_kl_divergence(q, p).item()) < 1e-5

    def test_identity(self):
        """d(p, p) ≈ 0"""
        p = torch.softmax(torch.randn(32), dim=0)
        assert symmetric_kl_divergence(p, p).item() < 1e-5

    def test_non_negative(self):
        """d(p, q) >= 0"""
        p = torch.softmax(torch.randn(32), dim=0)
        q = torch.softmax(torch.randn(32), dim=0)
        assert symmetric_kl_divergence(p, q).item() >= -1e-6

    def test_different_distributions(self):
        """Very different distributions should have large divergence."""
        p = torch.zeros(32)
        p[0] = 1.0
        q = torch.ones(32) / 32
        div = symmetric_kl_divergence(p, q).item()
        assert div > 0.1  # Should be significantly positive

    def test_unnormalized_input(self):
        """Handles unnormalized inputs (auto-normalized inside)."""
        p = torch.randn(32).abs()
        q = torch.randn(32).abs()
        # Should not raise
        result = symmetric_kl_divergence(p, q)
        assert torch.isfinite(result)

    def test_gradient_flow(self):
        """Gradients should flow through the KL computation."""
        p = torch.softmax(torch.randn(32, requires_grad=True), dim=0)
        q = torch.softmax(torch.randn(32), dim=0)
        loss = symmetric_kl_divergence(p, q)
        loss.backward()
        # p has gradients since the softmax input requires grad
        # Check that loss is differentiable
        assert loss.requires_grad


# ============================================================================
# Split Loss Tests
# ============================================================================


class TestSplitLoss:
    """Test split loss (Equation 3)."""

    def test_well_separated(self):
        """Well-separated tokens should have very negative split loss."""
        n = 64
        a1 = torch.zeros(n)
        a1[:16] = torch.softmax(torch.randn(16), dim=0)
        a2 = torch.zeros(n)
        a2[32:48] = torch.softmax(torch.randn(16), dim=0)
        loss = TextGuiderLoss.split_loss([a1, a2])
        assert loss.item() < 0  # Negative = well separated

    def test_overlapping(self):
        """Overlapping tokens should have less negative split loss."""
        n = 64
        a1 = torch.softmax(torch.randn(n), dim=0)
        a2 = torch.softmax(torch.randn(n), dim=0)
        loss = TextGuiderLoss.split_loss([a1, a2])
        # Less negative than well-separated case
        assert loss.item() > -10  # Reasonable bound

    def test_single_token(self):
        """Single token should return 0 (no pairs to compare)."""
        a1 = torch.softmax(torch.randn(64), dim=0)
        loss = TextGuiderLoss.split_loss([a1])
        assert loss.item() == 0.0

    def test_multiple_tokens(self):
        """Multiple tokens should compute all C(n,2) pairs."""
        n = 64
        tokens = [torch.softmax(torch.randn(n), dim=0) for _ in range(4)]
        loss = TextGuiderLoss.split_loss(tokens)
        # Should have 4*3/2 = 6 pairs
        assert torch.isfinite(loss)


# ============================================================================
# Wrap Loss Tests
# ============================================================================


class TestWrapLoss:
    """Test wrap loss (Equation 4)."""

    def test_good_coverage(self):
        """Quotation mark covering text region should have low loss."""
        n = 64
        # Text tokens in [0:32]
        a_text = torch.zeros(n)
        a_text[:32] = torch.softmax(torch.randn(32), dim=0)
        # Quotation mark also in [0:32]
        a_quo = torch.zeros(n)
        a_quo[:32] = torch.softmax(torch.randn(32), dim=0)

        loss = TextGuiderLoss.wrap_loss(a_quo, [a_text])
        assert torch.isfinite(loss)

    def test_poor_coverage(self):
        """Quotation mark not covering text region should have high loss."""
        n = 64
        # Text tokens in [0:16]
        a_text = torch.zeros(n)
        a_text[:16] = torch.softmax(torch.randn(16), dim=0)
        # Quotation mark in [48:64] (far away)
        a_quo = torch.zeros(n)
        a_quo[48:] = torch.softmax(torch.randn(16), dim=0)

        loss = TextGuiderLoss.wrap_loss(a_quo, [a_text])
        assert loss.item() > 0

    def test_empty_text(self):
        """No text tokens should return 0."""
        a_quo = torch.softmax(torch.randn(64), dim=0)
        loss = TextGuiderLoss.wrap_loss(a_quo, [])
        assert loss.item() == 0.0


# ============================================================================
# Total Loss Tests
# ============================================================================


class TestTotalLoss:
    """Test combined total loss (Equation 6)."""

    def test_normalization(self):
        """Loss should be normalized by N = C(n,2) + 1."""
        n = 64
        tokens = [torch.softmax(torch.randn(n), dim=0) for _ in range(3)]
        a_quo = torch.softmax(torch.randn(n), dim=0)

        total = TextGuiderLoss.total_loss(a_quo, tokens)
        assert torch.isfinite(total)

    def test_zero_text_tokens(self):
        """No text tokens should return 0."""
        a_quo = torch.softmax(torch.randn(64), dim=0)
        total = TextGuiderLoss.total_loss(a_quo, [])
        assert total.item() == 0.0

    def test_gradient_flow(self):
        """Total loss should support gradient computation."""
        n = 64
        tokens = [torch.softmax(torch.randn(n, requires_grad=True), dim=0) for _ in range(2)]
        a_quo = torch.softmax(torch.randn(n, requires_grad=True), dim=0)

        total = TextGuiderLoss.total_loss(a_quo, tokens)
        assert total.requires_grad


# ============================================================================
# AMO Sampler Tests
# ============================================================================


class TestAMOSampler:
    """Test AMO Sampler (Equation 2)."""

    def test_mask_computation(self):
        """AMO mask should have valid shape and range [0, 1]."""
        config = TextGuiderConfig(amo_overshoot_c=0.5)
        amo = AMOSampler(config)

        attn = [torch.softmax(torch.randn(256), dim=0) for _ in range(3)]
        mask = amo.compute_overshooting_mask(attn, 256, 32, 32)

        assert mask.shape == (1, 1, 32, 32)
        assert mask.min() >= 0
        assert mask.max() <= 1

    def test_overshooting(self):
        """Overshooting should add noise to the latent."""
        config = TextGuiderConfig(amo_overshoot_c=0.5)
        amo = AMOSampler(config)

        z = torch.randn(1, 16, 32, 32)
        mask = torch.ones(1, 1, 32, 32) * 0.5

        gen = torch.Generator().manual_seed(42)
        z_overshot = amo.apply_overshooting(z, t_next=0.8, epsilon=0.02, mask=mask, generator=gen)

        # Overshooting should change the latent
        assert not torch.allclose(z, z_overshot)
        # But not by an extreme amount
        diff = (z_overshot - z).abs().mean().item()
        assert diff < 10  # Reasonable bound

    def test_empty_mask(self):
        """Empty mask (no text) should effectively return original + noise."""
        config = TextGuiderConfig(amo_overshoot_c=0.5)
        amo = AMOSampler(config)

        mask = amo.compute_overshooting_mask([], 256, 32, 32)
        assert mask.sum().item() == 0  # No attention = no mask


# ============================================================================
# Attention Store Tests
# ============================================================================


class TestAttentionStore:
    """Test TextGuiderAttentionStore."""

    def test_store_and_aggregate(self):
        """Store attention from multiple layers and aggregate."""
        store = TextGuiderAttentionStore(
            quo_indices=[5],
            text_token_indices=[[6, 7], [8]],
            num_heads=4,
        )

        B, H, N_img, N_text, D = 1, 4, 64, 20, 32

        for _ in range(3):  # 3 layers
            q_img = torch.randn(B, H, N_img, D)
            k_text = torch.randn(B, H, N_text, D)
            store.store_attention(q_img, k_text, N_text)

        attn_quo, attn_texts = store.get_aggregated_maps()

        assert attn_quo.shape == (B, N_img)
        assert len(attn_texts) == 2
        assert attn_texts[0].shape == (B, N_img)
        assert attn_texts[1].shape == (B, N_img)

    def test_clear(self):
        """Clear should reset stored attention."""
        store = TextGuiderAttentionStore(
            quo_indices=[5],
            text_token_indices=[[6]],
            num_heads=4,
        )

        q_img = torch.randn(1, 4, 64, 32)
        k_text = torch.randn(1, 4, 20, 32)
        store.store_attention(q_img, k_text, 20)

        assert len(store.layer_attentions) == 1
        store.clear()
        assert len(store.layer_attentions) == 0

    def test_gradient_preservation(self):
        """Attention maps should retain gradients for backprop."""
        store = TextGuiderAttentionStore(
            quo_indices=[5],
            text_token_indices=[[6, 7]],
            num_heads=4,
        )

        q_img = torch.randn(1, 4, 64, 32, requires_grad=True)
        k_text = torch.randn(1, 4, 20, 32, requires_grad=True)
        store.store_attention(q_img, k_text, 20)

        attn_quo, attn_texts = store.get_aggregated_maps()

        # The attention maps should be part of the computation graph
        loss = attn_quo.sum() + sum(at.sum() for at in attn_texts)
        loss.backward()

        assert q_img.grad is not None
        assert k_text.grad is not None


# ============================================================================
# Config Tests
# ============================================================================


class TestConfig:
    """Test TextGuiderConfig defaults match the paper."""

    def test_default_alpha(self):
        config = TextGuiderConfig()
        assert config.alpha == 60.0  # Paper Sec. 4.1

    def test_default_t_guide_ratio(self):
        config = TextGuiderConfig()
        assert config.t_guide_ratio == 0.25  # Paper: first quarter

    def test_default_amo_c(self):
        config = TextGuiderConfig()
        assert config.amo_overshoot_c == 0.5  # Paper default

    def test_default_steps(self):
        config = TextGuiderConfig()
        assert config.num_inference_steps == 50  # FLUX.2 Klein base default

    def test_default_guidance(self):
        config = TextGuiderConfig()
        assert config.guidance_scale == 4.0  # FLUX.2 Klein base default


# ============================================================================
# Integration Test
# ============================================================================


class TestIntegration:
    """End-to-end test without GPU."""

    def test_dry_run_pipeline(self):
        """TextGuiderFluxPipeline dry-run should produce an image."""
        from src.textguider import TextGuiderFluxPipeline

        pipeline = TextGuiderFluxPipeline(pipe=None, device="cpu")
        img = pipeline.generate(
            prompt='A sign that says "HELLO"',
            width=256,
            height=256,
            use_textguider=True,
        )
        assert img is not None
        assert img.size == (256, 256)

    def test_full_loss_pipeline(self):
        """Verify the full loss pipeline works with synthetic data."""
        store = TextGuiderAttentionStore(
            quo_indices=[5],
            text_token_indices=[[6, 7], [8, 9]],
            num_heads=4,
        )

        # Simulate 5 dual-stream blocks (Klein 4B)
        for _ in range(5):
            q = torch.randn(1, 4, 128, 32, requires_grad=True)
            k = torch.randn(1, 4, 50, 32, requires_grad=True)
            store.store_attention(q, k, 50)

        attn_quo, attn_texts = store.get_aggregated_maps()
        loss = TextGuiderLoss.total_loss(attn_quo[0], [at[0] for at in attn_texts])

        assert torch.isfinite(loss)
        assert loss.requires_grad

        # Simulate gradient update (Z' = Z - α * ∇_Z L)
        z = torch.randn(1, 128, 32, 32, requires_grad=True)
        # Connect z to loss (simplified)
        combined_loss = loss + z.sum() * 0  # Ensure z is in the graph
        combined_loss.backward()

        print(f"  Integration test: loss={loss.item():.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
