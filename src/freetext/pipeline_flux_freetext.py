"""
FreeText Pipeline Integration for FLUX.2 [klein] 4B Base and Diffusers.
arXiv:2601.00535.
Provides plug-and-play inference with zero retraining or model modification.
"""

import os
import sys
from typing import List, Optional, Union, Dict, Any
from PIL import Image
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .sgmi import FreeTextConfig, SpectralGlyphInjector
from .glyph_renderer import extract_text_spans


class FreeTextFluxPipeline:
    """
    Wrapper for FLUX.2 [klein] Pipeline with FreeText enhancement.
    Works seamlessly with Diffusers pipelines or standalone invocation.
    """
    def __init__(
        self,
        pipe=None,
        config: Optional[FreeTextConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ):
        self.pipe = pipe
        self.config = config or FreeTextConfig()
        self.device = device
        self.dtype = dtype
        self.injector = SpectralGlyphInjector(config=self.config)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "black-forest-labs/FLUX.2-klein-base-4B",
        config: Optional[FreeTextConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        enable_cpu_offload: bool = True,
        **kwargs,
    ):
        """
        Loads pre-trained FLUX.2 pipeline and wraps it with FreeText.
        """
        try:
            from diffusers import Flux2KleinPipeline
            pipeline_cls = Flux2KleinPipeline
        except ImportError:
            try:
                from diffusers import FluxPipeline
                pipeline_cls = FluxPipeline
            except ImportError:
                pipeline_cls = None

        if pipeline_cls is None:
            print("[FreeText] Diffusers not installed or FLUX pipeline unavailable. Operating in mock/dry-run mode.")
            pipe = None
        else:
            print(f"[FreeText] Loading {model_id}...")
            pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
            if enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            elif hasattr(pipe, "to"):
                pipe.to(device)

        return cls(pipe=pipe, config=config, device=device, dtype=dtype)

    def create_step_callback(self, num_inference_steps: int):
        """
        Creates a Diffusers-compatible `callback_on_step_end` callback for FLUX sampling.
        """
        def callback_on_step_end(pipe, step: int, timestep: torch.Tensor, callback_kwargs: Dict[str, Any]):
            progress = (step + 1) / max(num_inference_steps, 1)
            latents = callback_kwargs.get("latents")
            if latents is not None:
                updated_latents = self.injector.inject_step(latents, progress=progress)
                callback_kwargs["latents"] = updated_latents
            return callback_kwargs

        return callback_on_step_end

    def generate(
        self,
        prompt: str,
        target_texts: Optional[List[str]] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = 42,
        use_freetext: bool = True,
        **kwargs,
    ) -> Image.Image:
        """
        Generates an image from prompt using FLUX.2 with optional FreeText enhancement.

        :param prompt: Text prompt describing the image (e.g. Banner khuyến mại 'PHỞ BÒ GIA TRUYỀN')
        :param target_texts: Optional list of specific Vietnamese text strings to enforce
        :param width: Output image width
        :param height: Output image height
        :param num_inference_steps: Number of denoising steps (50 for base, 4 for distilled)
        :param guidance_scale: Guidance scale (4.0 for base, 1.0 for distilled)
        :param generator: PyTorch generator for reproducibility
        :param seed: Seed for generation
        :param use_freetext: Whether to enable FreeText spectral glyph injection
        :return: PIL.Image result
        """
        if generator is None and seed is not None:
            gen_device = self.device if self.device != "cpu" and torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        # Configure FreeText target texts
        if target_texts:
            self.config.override_texts = target_texts
        self.config.enabled = use_freetext

        # If running with live Diffusers pipeline
        if self.pipe is not None:
            vae = getattr(self.pipe, "vae", None)
            if use_freetext and vae is not None:
                has_text = self.injector.prepare(
                    prompt=prompt,
                    vae=vae,
                    height=height,
                    width=width,
                    device=self.device,
                    dtype=self.dtype,
                )
                if has_text:
                    print(f"[FreeText] Active: Targets={self.config.override_texts or extract_text_spans(prompt)}")

            callback = self.create_step_callback(num_inference_steps) if use_freetext else None

            pipe_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                **kwargs,
            }
            if callback is not None:
                pipe_kwargs["callback_on_step_end"] = callback
                pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            result = self.pipe(**pipe_kwargs)
            return result.images[0]

        # Dry-run / CPU simulation mode (creates demonstration canvas for testing pipeline)
        print(f"[FreeText Dry-Run] Simulating generation for prompt: {prompt[:60]}... (use_freetext={use_freetext})")
        img, _, _ = self.injector.renderer.render_text_canvas(
            texts=target_texts or extract_text_spans(prompt) or ["Demo Vietnamese Text"],
            width=width,
            height=height,
            bg_color=(25, 28, 36) if not use_freetext else (20, 20, 20),
            text_color=(180, 180, 180) if not use_freetext else (255, 215, 0),
        )
        return img
