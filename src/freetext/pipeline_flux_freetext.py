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

    @property
    def is_live(self) -> bool:
        """Whether a real Diffusers pipeline is attached (not CPU dry-run)."""
        return self.pipe is not None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "black-forest-labs/FLUX.2-klein-base-4B",
        config: Optional[FreeTextConfig] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        torch_dtype: Optional[torch.dtype] = None,
        enable_cpu_offload: bool = True,
        **kwargs,
    ):
        """
        Loads pre-trained FLUX.2 pipeline and wraps it with FreeText.
        """
        # Resolve at call time. Resolving this in the function signature can
        # capture a stale CUDA availability value in notebook/Accelerate jobs.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_dtype = torch_dtype or dtype
        if resolved_dtype is None:
            resolved_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Avoid duplicate kwargs
        kwargs.pop("torch_dtype", None)
        kwargs.pop("dtype", None)

        # FLUX.2 Klein has a different VAE/latent packing and text encoder from
        # FLUX.1. Falling back to FluxPipeline can appear to work while silently
        # producing incompatible latents, so only load the native class here.
        try:
            from diffusers import Flux2KleinPipeline
            pipeline_cls = Flux2KleinPipeline
        except ImportError:
            pipeline_cls = None

        if pipeline_cls is None:
            print("[FreeText] Diffusers not installed or FLUX pipeline unavailable. Operating in mock/dry-run mode.")
            pipe = None
        else:
            print(f"[FreeText] Loading {model_id} (dtype={resolved_dtype})...")
            pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=resolved_dtype, **kwargs)
            if enable_cpu_offload and device.startswith("cuda") and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            elif hasattr(pipe, "to"):
                pipe.to(device)

        return cls(pipe=pipe, config=config, device=device, dtype=resolved_dtype)

    def create_step_callback(self, num_inference_steps: int, attention_state: Optional[Dict[str, Any]] = None):
        """
        Creates a Diffusers-compatible `callback_on_step_end` callback for FLUX sampling.
        """
        def callback_on_step_end(pipe, step: int, timestep: torch.Tensor, callback_kwargs: Dict[str, Any]):
            timestep_sigma = None
            scheduler = getattr(pipe, "scheduler", None)
            sigmas = getattr(scheduler, "sigmas", None)
            if sigmas is not None and len(sigmas) > 0:
                # FlowMatchEulerDiscreteScheduler exposes the actual sigma
                # trajectory; use it instead of assuming uniformly spaced
                # denoising steps.
                sigma_index = min(int(step), len(sigmas) - 1)
                timestep_sigma = float(sigmas[sigma_index].detach().cpu().item())
                progress = max(0.0, min(1.0, 1.0 - timestep_sigma))
            else:
                progress = (step + 1) / max(num_inference_steps, 1)
            if attention_state is not None:
                self.injector.localization.finalize_attention_step(attention_state["recorder"], step)
                if not attention_state["locked"] and progress >= attention_state["lock_progress"]:
                    attention_mask = self.injector.localization.build_attention_mask(
                        regions=self.injector.regions,
                        target_h=self.injector.z0_glyph.shape[-2],
                        target_w=self.injector.z0_glyph.shape[-1],
                        img_w=self.injector.img_width,
                        img_h=self.injector.img_height,
                        device=self.injector.z0_glyph.device,
                        dtype=self.injector.z0_glyph.dtype,
                    )
                    if attention_mask is not None:
                        self.injector.mask = attention_mask
                        attention_state["locked"] = True
                        print("[FreeText] Attention localization locked; using top-k refined mask.")
            latents = callback_kwargs.get("latents")
            if latents is not None:
                updated_latents = self.injector.inject_step(
                    latents,
                    progress=progress,
                    timestep_sigma=timestep_sigma,
                )
                callback_kwargs["latents"] = updated_latents
            return callback_kwargs

        return callback_on_step_end

    def _get_execution_device(self):
        """Return the device used by Diffusers/Accelerate for model execution."""
        if self.pipe is not None:
            execution_device = getattr(self.pipe, "_execution_device", None)
            if execution_device is not None:
                return execution_device
        return self.device

    def generate(
        self,
        prompt: str,
        target_texts: Optional[List[str]] = None,
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
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
        # Do not leak an explicit text list into the next request.
        self.config.override_texts = target_texts
        self.config.enabled = use_freetext

        # If running with live Diffusers pipeline
        if self.pipe is not None:
            vae = getattr(self.pipe, "vae", None)
            attention_state = None
            capture_handle = None
            if use_freetext and vae is not None:
                has_text = self.injector.prepare(
                    prompt=prompt,
                    vae=vae,
                    height=height,
                    width=width,
                    # With Accelerate CPU offload, ``_execution_device`` may
                    # still report CPU immediately before the VAE hook moves
                    # the module to CUDA. Create the glyph tensor on the
                    # configured inference device to avoid CPU/CUDA mismatch.
                    device=self.device,
                    dtype=self.dtype,
                )
                if has_text:
                    print(f"[FreeText] Active: Targets={self.config.override_texts or extract_text_spans(prompt)}")

                    if self.config.localization_mode == "attention":
                        tokenizer = getattr(self.pipe, "tokenizer", None)
                        transformer = getattr(self.pipe, "transformer", None)
                        if tokenizer is not None and transformer is not None:
                            try:
                                capture_handle, recorder = self.injector.localization.install_flux2_capture(
                                    transformer,
                                    tokenizer,
                                    prompt,
                                    self.config.override_texts or extract_text_spans(prompt),
                                    image_height=height,
                                    image_width=width,
                                )
                                attention_state = {
                                    "recorder": recorder,
                                    "locked": False,
                                    "lock_progress": min(max(self.config.localization_warmup_ratio, 0.05), 0.8),
                                }
                                print("[FreeText] Attention localization capture enabled.")
                            except Exception as exc:
                                print(f"[FreeText] Attention capture unavailable; using layout mask: {exc}")

            callback = self.create_step_callback(num_inference_steps, attention_state) if use_freetext else None

            pipe_kwargs = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                **kwargs,
            }
            if image is not None:
                if "image" in pipe_kwargs:
                    raise ValueError("Pass the edit image either via image= or kwargs, not both")
                pipe_kwargs["image"] = image
            if callback is not None:
                # Diffusers versions differ in the callback tensor allow-list.
                # Only pass latents when the installed Klein pipeline exposes it.
                callback_inputs = getattr(self.pipe, "_callback_tensor_inputs", ())
                if "latents" in callback_inputs:
                    pipe_kwargs["callback_on_step_end"] = callback
                    pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                else:
                    print("[FreeText] Warning: pipeline does not expose callback latents; running base sampling.")

            try:
                result = self.pipe(**pipe_kwargs)
                return result.images[0]
            finally:
                if capture_handle is not None:
                    capture_handle.close(getattr(self.pipe, "transformer", None))

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
