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
    # FreeText uses t_start=0.8T -> t_end=0.6T in descending diffusion time.
    # With progress increasing from pure noise to the clean image, this is
    # approximately the [0.20, 0.40] window.
    t_start: float = 0.20
    t_end: float = 0.40
    # Eq. (13) defines lambda's peak as 1.0; callers may lower this for an
    # application-specific strength ablation.
    injection_strength: float = 1.0
    center_freq: float = 0.22    # Log-Gabor center frequency rho_0
    bandwidth_ratio: float = 0.55 # Log-Gabor bandwidth ratio
    font_path: Optional[str] = None
    override_texts: Optional[List[str]] = None  # Explicit text list (if not extracted from prompt)
    localization_mode: str = "attention"  # "attention" or "layout"
    localization_warmup_ratio: float = 0.30
    top_k_attention_pairs: int = 4


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
    Unpacks 2x2 packed tokens into 4D spatial latents.

    FLUX.1 commonly uses [B, N, 64] / 16 channels, while FLUX.2 Klein 4B
    uses [B, N, 128] / 32 channels. Infer the channel count from the token
    width instead of assuming the FLUX.1 layout.
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


def patchify_vae_latents(latents: torch.Tensor) -> torch.Tensor:
    """Match Flux2's VAE patchification before BatchNorm normalization."""
    B, C, H, W = latents.shape
    if H % 2 or W % 2:
        raise ValueError(f"VAE latent spatial size must be even, got {H}x{W}")
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(B, C * 4, H // 2, W // 2)


def unpatchify_vae_latents(latents: torch.Tensor) -> torch.Tensor:
    """Convert normalized Flux2 VAE patches back to the denoiser layout."""
    B, C, H, W = latents.shape
    if C % 4:
        raise ValueError(f"Patchified VAE channels must be divisible by 4, got {C}")
    latents = latents.reshape(B, C // 4, 2, 2, H, W)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(B, C // 4, H * 2, W * 2)


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
        self.localization = AttentionLocalization(top_k_pairs=self.config.top_k_attention_pairs)
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
        # Each request must select localization maps from its own prompt and
        # denoising trajectory, especially when one pipeline serves a batch
        # of comparison prompts.
        self.localization.clear()

        texts = self.config.override_texts or extract_text_spans(prompt)
        if not texts:
            return False

        # Keep the caller's execution device. With Accelerate CPU offload, a
        # VAE parameter may temporarily report CPU even though its hook moves
        # the module to CUDA for encode(); using that transient device causes
        # CPU-input/CUDA-weight mismatches.
        target_device = device
        # Accelerate CPU-offload keeps the module on CPU between calls, while
        # its hook moves it to the GPU immediately before ``forward``. Use the
        # hook's execution device when available; otherwise the wrapper device
        # is the correct default.
        if vae is not None:
            for module in (vae, getattr(vae, "encoder", None)):
                hook = getattr(module, "_hf_hook", None)
                execution_device = getattr(hook, "execution_device", None)
                if execution_device is not None:
                    target_device = execution_device
                    break
        target_dtype = dtype
        if vae is not None and hasattr(vae, "parameters"):
            try:
                p = next(vae.parameters())
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
                try:
                    encoded = vae.encode(glyph_img_tensor)
                except RuntimeError as exc:
                    # Some Accelerate versions install the hook only on the
                    # first VAE forward. Retry once on the hook execution
                    # device for the known CPU/CUDA input mismatch.
                    message = str(exc)
                    if "CPUBFloat16Type" not in message and "CPUFloatType" not in message:
                        raise
                    retry_device = None
                    for module in (vae, getattr(vae, "encoder", None)):
                        hook = getattr(module, "_hf_hook", None)
                        retry_device = getattr(hook, "execution_device", None)
                        if retry_device is not None:
                            break
                    if retry_device is None or str(retry_device) == "cpu":
                        raise
                    glyph_img_tensor = glyph_img_tensor.to(device=retry_device)
                    encoded = vae.encode(glyph_img_tensor)
                if hasattr(encoded, "latent_dist"):
                    # Flux2's Klein pipeline uses the deterministic mode,
                    # patchifies it, applies VAE BatchNorm statistics, and
                    # then feeds the normalized patches to the transformer.
                    self.z0_glyph = (
                        encoded.latent_dist.mode()
                        if hasattr(encoded.latent_dist, "mode")
                        else encoded.latent_dist.sample()
                    )
                elif hasattr(encoded, "latents"):
                    self.z0_glyph = encoded.latents
                elif isinstance(encoded, tuple):
                    self.z0_glyph = encoded[0]
                else:
                    self.z0_glyph = encoded

                if hasattr(vae, "bn") and self.z0_glyph.ndim == 4:
                    patchified = patchify_vae_latents(self.z0_glyph)
                    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
                        patchified.device, patchified.dtype
                    )
                    bn_std = torch.sqrt(
                        vae.bn.running_var.view(1, -1, 1, 1).to(
                            patchified.device, patchified.dtype
                        )
                        + getattr(getattr(vae, "config", None), "batch_norm_eps", 1e-5)
                    )
                    self.z0_glyph = unpatchify_vae_latents((patchified - bn_mean) / bn_std)
                else:
                    # Compatibility path for non-Flux2 VAEs.
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

        # Paper Eq. (13): decay from full injection at the start of the
        # mid-early window to zero at its end.
        norm_t = (progress - t_start) / (t_end - t_start)
        weight = 0.5 * (1.0 + math.cos(math.pi * norm_t)) * self.config.injection_strength
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
