"""
Spectral-Modulated Glyph Injection (SGMI) Core Engine.
arXiv:2601.00535 Section 3.2.
Integrates VAE latent projection, Flow noise alignment, Log-Gabor spectral modulation,
and cosine-annealed spatiotemporal replacement.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import torch

from .glyph_renderer import GlyphRenderer, extract_text_spans
from .log_gabor import LogGaborFilter
from .localization import AttentionLocalization


@dataclass
class FreeTextConfig:
    """Configuration hyper-parameters for FreeText."""
    enabled: bool = True
    t_start: float = 0.20        # Start of injection window (normalized time [0, 1])
    t_end: float = 0.75          # End of injection window
    injection_strength: float = 0.85  # Peak injection weight lambda
    center_freq: float = 0.22    # Log-Gabor center frequency rho_0
    bandwidth_ratio: float = 0.55 # Log-Gabor bandwidth ratio
    font_path: Optional[str] = None
    override_texts: Optional[List[str]] = None  # Explicit text list (if not extracted from prompt)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Packs 4D spatial latents [B, 16, H, W] into FLUX 3D packed tokens [B, (H//2)*(W//2), 64].
    """
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
    return latents


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Unpacks FLUX 3D packed tokens [B, N, 64] into 4D spatial latents [B, 16, H_lat, W_lat].
    """
    B, N, D = latents.shape
    H_lat = height // 8
    W_lat = width // 8
    H_p = H_lat // 2
    W_p = W_lat // 2
    latents = latents.view(B, H_p, W_p, 16, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(B, 16, H_lat, W_lat)
    return latents


class SpectralGlyphInjector:
    """
    Orchestrates the FreeText injection process for FLUX.2 and other DiT models.
    """
    def __init__(self, config: Optional[FreeTextConfig] = None):
        self.config = config or FreeTextConfig()
        self.renderer = GlyphRenderer(font_path=self.config.font_path)
        self.filter = LogGaborFilter(
            center_freq=self.config.center_freq,
            bandwidth_ratio=self.config.bandwidth_ratio,
        )
        self.localization = AttentionLocalization()
        self.z0_glyph: Optional[torch.Tensor] = None
        self.mask: Optional[torch.Tensor] = None
        self.regions: List[Dict] = []
        self.noise_ref: Optional[torch.Tensor] = None
        self.img_height: int = 1024
        self.img_width: int = 1024

    def prepare(
        self,
        prompt: str,
        vae,
        height: int = 1024,
        width: int = 1024,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> bool:
        """
        Extracts texts from prompt, renders the Vietnamese glyph canvas,
        encodes it into latent space via VAE, and creates the spatial mask.

        :return: True if text targets were found and prepared, False otherwise.
        """
        self.img_height = height
        self.img_width = width

        texts = self.config.override_texts or extract_text_spans(prompt)
        if not texts:
            return False

        # Determine target device and dtype from VAE if available
        target_device = device
        target_dtype = dtype
        if vae is not None and hasattr(vae, "parameters"):
            try:
                p = next(vae.parameters())
                target_device = p.device
                target_dtype = p.dtype
            except StopIteration:
                pass

        # 1. Render glyph canvas
        glyph_img_tensor, mask_pixel_tensor, self.regions = self.renderer.get_glyph_tensor(
            texts=texts,
            width=width,
            height=height,
            device=target_device,
            dtype=target_dtype,
        )

        # 2. Encode glyph image with VAE to get z0_glyph
        with torch.no_grad():
            if vae is not None and hasattr(vae, "encode"):
                encoded = vae.encode(glyph_img_tensor)
                if hasattr(encoded, "latent_dist"):
                    self.z0_glyph = encoded.latent_dist.sample()
                elif hasattr(encoded, "latents"):
                    self.z0_glyph = encoded.latents
                elif isinstance(encoded, tuple):
                    self.z0_glyph = encoded[0]
                else:
                    self.z0_glyph = encoded

                # Scale latent if scaling_factor is present
                scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", 0.3611)
                shift_factor = getattr(getattr(vae, "config", None), "shift_factor", 0.1159)
                self.z0_glyph = (self.z0_glyph - shift_factor) * scaling_factor
            else:
                # Mock / Dry-run mode: construct latent tensor directly
                latent_h = height // 8
                latent_w = width // 8
                self.z0_glyph = torch.randn((1, 16, latent_h, latent_w), device=target_device, dtype=target_dtype)

        # 3. Create latent-resolution binary mask
        lat_h, lat_w = self.z0_glyph.shape[-2], self.z0_glyph.shape[-1]
        self.mask = self.localization.create_layout_mask(
            regions=self.regions,
            latent_h=lat_h,
            latent_w=lat_w,
            img_w=width,
            img_h=height,
            device=target_device,
            dtype=target_dtype,
        )

        # Fixed reference noise for reproducibility during step trajectory
        self.noise_ref = torch.randn_like(self.z0_glyph)
        return True

    def compute_annealed_weight(self, progress: float) -> float:
        """
        Computes cosine-annealed injection weight w(t) for progress in [0, 1].
        progress: 0.0 (start of denoising / pure noise) to 1.0 (end of denoising / clean image).
        """
        t_start = self.config.t_start
        t_end = self.config.t_end

        if progress < t_start or progress > t_end:
            return 0.0

        # Cosine curve peaking in middle of injection window
        norm_t = (progress - t_start) / (t_end - t_start)
        weight = 0.5 * (1.0 + math.cos((norm_t - 0.5) * 2.0 * math.pi)) * self.config.injection_strength
        return float(weight)

    def inject_step(
        self,
        latents: torch.Tensor,
        progress: float,
        timestep_sigma: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Performs SGMI injection at current denoising step.
        Handles both 4D spatial latents [B, 16, H, W] and FLUX 3D packed tokens [B, N, 64].

        :param latents: Current model latents
        :param progress: Denoising progress in [0.0, 1.0]
        :param timestep_sigma: Noise level sigma (for Flow Matching, sigma = 1 - progress)
        :return: Updated latents with spectral glyph guidance
        """
        if not self.config.enabled or self.z0_glyph is None or self.mask is None:
            return latents

        weight = self.compute_annealed_weight(progress)
        if weight <= 1e-4:
            return latents

        is_3d_packed = (latents.ndim == 3)

        # If FLUX 3D packed format [B, N, 64], unpack to 4D [B, 16, H_lat, W_lat]
        if is_3d_packed:
            latents_4d = unpack_latents(latents, height=self.img_height, width=self.img_width)
        else:
            latents_4d = latents

        cur_device = latents_4d.device
        cur_dtype = latents_4d.dtype

        # Ensure z0_glyph and noise_ref match current device and dtype
        z0 = self.z0_glyph.to(device=cur_device, dtype=cur_dtype)
        noise = self.noise_ref.to(device=cur_device, dtype=cur_dtype)
        mask = self.mask.to(device=cur_device, dtype=cur_dtype)

        # Expand batch size if needed
        if z0.shape[0] != latents_4d.shape[0]:
            z0 = z0.repeat(latents_4d.shape[0], 1, 1, 1)
            noise = noise.repeat(latents_4d.shape[0], 1, 1, 1)
            mask = mask.repeat(latents_4d.shape[0], 1, 1, 1)

        # Noise level for flow matching: z_ref(t) = (1 - sigma) * z0 + sigma * eps
        sigma = timestep_sigma if timestep_sigma is not None else (1.0 - progress)
        z_ref_t = (1.0 - sigma) * z0 + sigma * noise

        # Apply 2D Log-Gabor spectral modulation
        z_sgmi = self.filter.apply_spectral_modulation(z_ref_t).to(device=cur_device, dtype=cur_dtype)

        # Annealed replacement: z(t) = (1 - w * M) * z(t) + (w * M) * z_sgmi
        w_mask = weight * mask
        updated_4d = (1.0 - w_mask) * latents_4d + w_mask * z_sgmi

        # Pack back to 3D if original input was packed
        if is_3d_packed:
            return pack_latents(updated_4d)
        return updated_4d
