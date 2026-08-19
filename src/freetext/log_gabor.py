"""
Log-Gabor 2D Spectral Filter for FreeText (Spectral-Modulated Glyph Injection - SGMI).
arXiv:2601.00535 Section 3.2.2.
"""

import math
import torch
import torch.fft


class LogGaborFilter:
    """
    2D Log-Gabor Band-pass Filter implemented in PyTorch.
    Enhances mid-to-high frequency components corresponding to glyph structures
    while suppressing low-frequency semantic background and extreme high-frequency noise.
    """
    def __init__(
        self,
        center_freq: float = 0.22,
        bandwidth_ratio: float = 0.55,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param center_freq: rho_0, center frequency normalized to [0, 0.5]
        :param bandwidth_ratio: sigma_rho / rho_0 (ratio controlling frequency bandwidth)
        """
        self.center_freq = max(1e-4, center_freq)
        self.bandwidth_ratio = max(1e-4, bandwidth_ratio)
        self.device = device
        self.dtype = dtype
        self._filter_cache = {}

    def construct_2d_filter(self, height: int, width: int, device=None, dtype=None) -> torch.Tensor:
        """
        Constructs the 2D Log-Gabor filter grid centered at (H/2, W/2).
        Output shape: [1, 1, H, W]
        """
        key = (height, width, str(device), str(dtype))
        if key in self._filter_cache:
            return self._filter_cache[key]

        dev = device or self.device
        dt = dtype or self.dtype

        # Coordinate grid normalized to [-0.5, 0.5]
        y = torch.linspace(-0.5, 0.5, steps=height, device=dev, dtype=dt)
        x = torch.linspace(-0.5, 0.5, steps=width, device=dev, dtype=dt)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        # Radial frequency rho
        radius = torch.sqrt(grid_x**2 + grid_y**2)
        radius[radius == 0] = 1e-6  # Avoid log(0) at DC component

        # Log-Gabor radial transfer function
        # G(rho) = exp( - (ln(rho / rho_0))^2 / (2 * (ln(sigma_rho / rho_0))^2) )
        log_term = torch.log(radius / self.center_freq)
        sigma_term = math.log(self.bandwidth_ratio)
        if abs(sigma_term) < 1e-6:
            sigma_term = 0.55

        denom = 2.0 * (sigma_term ** 2)
        gabor_2d = torch.exp(-(log_term ** 2) / denom)

        # Set DC component to 0 (suppress low frequency DC drift)
        center_h, center_w = height // 2, width // 2
        gabor_2d[center_h, center_w] = 0.0

        # Normalize filter maximum to 1.0
        max_val = gabor_2d.max()
        if max_val > 0:
            gabor_2d = gabor_2d / max_val

        # Reshape to [1, 1, H, W]
        gabor_2d = gabor_2d.unsqueeze(0).unsqueeze(0)
        self._filter_cache[key] = gabor_2d
        return gabor_2d

    def apply_spectral_modulation(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D FFT, Log-Gabor frequency filtering, and 2D IFFT on input latent.

        :param latent: Tensor of shape [B, C, H, W]
        :return: Modulated latent tensor of same shape [B, C, H, W]
        """
        B, C, H, W = latent.shape
        orig_dtype = latent.dtype
        orig_device = latent.device

        # Perform FFT in float32 for numerical stability
        latent_f32 = latent.to(dtype=torch.float32)

        # 2D FFT with DC shifted to center
        fft_latent = torch.fft.fftshift(torch.fft.fft2(latent_f32, dim=(-2, -1)), dim=(-2, -1))

        # Build filter
        filter_2d = self.construct_2d_filter(H, W, device=orig_device, dtype=torch.float32)

        # Frequency domain multiplication
        filtered_fft = fft_latent * filter_2d

        # Inverse FFT
        ifft_shift = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
        reconstructed = torch.fft.ifft2(ifft_shift, dim=(-2, -1)).real

        return reconstructed.to(device=orig_device, dtype=orig_dtype)
