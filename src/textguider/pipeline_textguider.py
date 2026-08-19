"""
TextGuider Pipeline for FLUX.2 Klein 4B Base.
arXiv:2512.09350

Integrates TextGuider latent guidance with AMO Sampler for training-free
text rendering improvement. Works with Diffusers' Flux2KleinPipeline.

Pipeline flow (per denoising step):
  1. For guided steps (first t_guide fraction):
     a. Enable gradients on Z_{t_k}
     b. Forward pass through transformer (dual-stream blocks with capture)
     c. Compute cross-modal attention maps A_{τ_quo}, A_{τ_text}
     d. Compute TextGuider loss L = (L_split + L_wrap) / N
     e. Backprop: ∇_{Z_{t_k}} L
     f. Update: Z'_{t_k} = Z_{t_k} - α * ∇_{Z_{t_k}} L
     g. Standard Euler step with velocity prediction
     h. AMO overshooting on the text region
  2. For remaining steps:
     a. Standard Euler step
     b. AMO overshooting (using last attention mask)
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .textguider import (
    TextGuiderConfig,
    TextGuiderLoss,
    TextGuiderTokenParser,
    AMOSampler,
)
from .textguider_attention import (
    TextGuiderAttentionStore,
    TextGuiderForwardWrapper,
)


class TextGuiderFluxPipeline:
    """FLUX.2 Klein 4B Base pipeline with TextGuider text rendering enhancement.

    Combines Diffusers' Flux2KleinPipeline with:
      - TextGuider latent guidance (split loss + wrap loss)
      - AMO Sampler overshooting
    for training-free text rendering improvement.

    Example usage:
        pipeline = TextGuiderFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-base-4B"
        )
        image = pipeline.generate(
            prompt='A poster with "GIẢM GIÁ 50%" written on it',
            seed=42,
        )
    """

    def __init__(
        self,
        pipe=None,
        config: Optional[TextGuiderConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.pipe = pipe
        self.config = config or TextGuiderConfig()
        self.device = device
        self.dtype = dtype

        # AMO Sampler
        self.amo_sampler = AMOSampler(self.config) if self.config.amo_enabled else None

        # Last attention mask for AMO (persists across steps)
        self._amo_mask: Optional[Tensor] = None

    @property
    def is_live(self) -> bool:
        """Whether a real Diffusers pipeline is attached."""
        return self.pipe is not None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "black-forest-labs/FLUX.2-klein-base-4B",
        config: Optional[TextGuiderConfig] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        torch_dtype: Optional[torch.dtype] = None,
        enable_cpu_offload: bool = True,
        **kwargs,
    ):
        """Load FLUX.2 Klein base model with TextGuider enhancement."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved_dtype = torch_dtype or dtype
        if resolved_dtype is None:
            resolved_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        kwargs.pop("torch_dtype", None)
        kwargs.pop("dtype", None)

        try:
            from diffusers import Flux2KleinPipeline
            pipeline_cls = Flux2KleinPipeline
        except ImportError:
            pipeline_cls = None

        if pipeline_cls is None:
            print("[TextGuider] Diffusers Flux2KleinPipeline unavailable. Operating in dry-run mode.")
            pipe = None
        else:
            print(f"[TextGuider] Loading {model_id} (dtype={resolved_dtype})...")
            pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=resolved_dtype, **kwargs)
            if enable_cpu_offload and device.startswith("cuda") and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            elif hasattr(pipe, "to"):
                pipe.to(device)

        return cls(pipe=pipe, config=config, device=device, dtype=resolved_dtype)

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = 42,
        generator: Optional[torch.Generator] = None,
        use_textguider: bool = True,
        **kwargs,
    ) -> Image.Image:
        """Generate an image with TextGuider-enhanced text rendering.

        Args:
            prompt: Text prompt. Text to render should be in quotation marks,
                e.g. 'A sign that says "HELLO WORLD"'.
            width: Output width.
            height: Output height.
            num_inference_steps: Denoising steps (default: config value).
            guidance_scale: CFG scale (default: config value).
            seed: Random seed.
            generator: Optional PyTorch generator.
            use_textguider: Whether to enable TextGuider guidance.
            **kwargs: Additional args passed to the base pipeline.

        Returns:
            PIL.Image result.
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale

        if generator is None and seed is not None:
            gen_device = self.device if self.device != "cpu" and torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        if self.pipe is None:
            return self._dry_run(prompt, width, height)

        # --- Parse tokens ---
        token_info = None
        if use_textguider:
            tokenizer = getattr(self.pipe, "tokenizer", None)
            if tokenizer is not None:
                token_info = TextGuiderTokenParser.parse_tokens(tokenizer, prompt)
                if token_info["num_text_tokens"] > 0:
                    print(
                        f"[TextGuider] Parsed tokens: "
                        f"quo={token_info['quo_indices']}, "
                        f"text_groups={[len(g) for g in token_info['text_token_indices']]}, "
                        f"strings={token_info['text_strings']}"
                    )
                else:
                    print("[TextGuider] No quoted text found in prompt. Running baseline.")
                    use_textguider = False

        if not use_textguider or token_info is None or token_info["num_text_tokens"] == 0:
            # Baseline generation without TextGuider
            return self._baseline_generate(
                prompt, width, height, num_inference_steps, guidance_scale, generator, **kwargs
            )

        # --- TextGuider-enhanced generation ---
        t_guide_steps = max(1, int(num_inference_steps * self.config.t_guide_ratio))
        print(
            f"[TextGuider] Config: α={self.config.alpha}, "
            f"t_guide={t_guide_steps}/{num_inference_steps} steps, "
            f"AMO={'on' if self.config.amo_enabled else 'off'} "
            f"(c={self.config.amo_overshoot_c})"
        )

        # Create callback for TextGuider guidance
        callback = self._create_textguider_callback(
            token_info, t_guide_steps, num_inference_steps, height, width, prompt
        )

        pipe_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            **kwargs,
        }

        # Set up callback
        callback_inputs = getattr(self.pipe, "_callback_tensor_inputs", ())
        if "latents" in callback_inputs:
            pipe_kwargs["callback_on_step_end"] = callback
            pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        else:
            print("[TextGuider] Warning: pipeline does not expose callback latents; attempting full integration.")
            return self._full_textguider_generate(
                prompt, width, height, num_inference_steps, guidance_scale,
                generator, token_info, t_guide_steps, **kwargs
            )

        result = self.pipe(**pipe_kwargs)
        return result.images[0]

    def _create_textguider_callback(
        self,
        token_info: Dict,
        t_guide_steps: int,
        total_steps: int,
        height: int,
        width: int,
        prompt: str,
    ):
        """Create a Diffusers callback_on_step_end for TextGuider guidance.

        This callback:
        1. For guided steps: computes TextGuider loss, updates latents
        2. For all steps with AMO: applies overshooting
        """
        transformer = getattr(self.pipe, "transformer", None)
        if transformer is None:
            raise RuntimeError("Pipeline has no transformer attribute")

        num_heads = getattr(transformer.config, "num_attention_heads", 24)
        store = TextGuiderAttentionStore(
            quo_indices=token_info["quo_indices"],
            text_token_indices=token_info["text_token_indices"],
            num_heads=num_heads,
        )
        wrapper = TextGuiderForwardWrapper(
            model=transformer,
            store=store,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

        lat_h = height // 8
        lat_w = width // 8

        # Pre-encode prompt embeddings & IDs so they are directly available for gradient tracking
        prompt_embeds = None
        txt_ids = None
        if hasattr(self.pipe, "encode_prompt"):
            try:
                res = self.pipe.encode_prompt(prompt=prompt, prompt_2=None)
                if isinstance(res, tuple):
                    prompt_embeds = res[0]
                    txt_ids = res[1] if len(res) > 1 else None
            except Exception as exc:
                print(f"[TextGuider] Note: encode_prompt pre-encoding: {exc}")

        img_ids = None
        if hasattr(self.pipe, "_prepare_latent_image_ids"):
            try:
                exec_dev = getattr(self.pipe, "_execution_device", transformer.device)
                img_ids = self.pipe._prepare_latent_image_ids(
                    1, lat_h // 2, lat_w // 2, exec_dev, transformer.dtype
                )
            except Exception:
                pass

        def callback_on_step_end(pipe, step: int, timestep: Tensor, callback_kwargs: Dict):
            latents = callback_kwargs.get("latents")
            if latents is None:
                return callback_kwargs

            is_guided_step = step < t_guide_steps

            if is_guided_step:
                # --- TextGuider Latent Guidance ---
                latents = self._apply_textguider_guidance(
                    latents=latents,
                    transformer=transformer,
                    wrapper=wrapper,
                    store=store,
                    pipe=pipe,
                    step=step,
                    timestep=timestep,
                    lat_h=lat_h,
                    lat_w=lat_w,
                    token_info=token_info,
                    prompt_embeds=prompt_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                )
                callback_kwargs["latents"] = latents

            # --- AMO Overshooting ---
            if self.config.amo_enabled and self.amo_sampler is not None and self._amo_mask is not None:
                scheduler = getattr(pipe, "scheduler", None)
                sigmas = getattr(scheduler, "sigmas", None)
                if sigmas is not None and step + 1 < len(sigmas):
                    sigma_next = float(sigmas[step + 1].item())
                    sigma_curr = float(sigmas[step].item()) if step < len(sigmas) else sigma_next
                    epsilon = sigma_next - sigma_curr

                    latents_4d = self._unpack_to_4d(latents, height, width)
                    latents_4d = self.amo_sampler.apply_overshooting(
                        latents_4d,
                        t_next=sigma_next,
                        epsilon=abs(epsilon),
                        mask=self._amo_mask,
                    )
                    callback_kwargs["latents"] = self._pack_to_3d(latents_4d)

            return callback_kwargs

        return callback_on_step_end

    def _apply_textguider_guidance(
        self,
        latents: Tensor,
        transformer,
        wrapper: TextGuiderForwardWrapper,
        store: TextGuiderAttentionStore,
        pipe,
        step: int,
        timestep: Tensor,
        lat_h: int,
        lat_w: int,
        token_info: Dict,
        prompt_embeds: Optional[Tensor] = None,
        txt_ids: Optional[Tensor] = None,
        img_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply TextGuider latent guidance: Z' = Z - α * ∇_Z L."""
        original_latents = latents.detach().clone()
        latents_grad = latents.detach().clone().requires_grad_(True)

        try:
            # Fallback to pipeline internals if prompt_embeds was not precomputed
            enc_states = prompt_embeds if prompt_embeds is not None else getattr(pipe, "_current_encoder_hidden_states", None)
            t_ids = txt_ids if txt_ids is not None else getattr(pipe, "_current_txt_ids", None)
            i_ids = img_ids if img_ids is not None else getattr(pipe, "_current_img_ids", None)

            store.clear()
            attn_quo, attn_texts = wrapper.compute_attention_maps_diffusers(
                transformer=transformer,
                latents=latents_grad,
                encoder_hidden_states=enc_states,
                timestep=timestep,
                img_ids=i_ids,
                txt_ids=t_ids,
                guidance=None,
                joint_attention_kwargs=None,
            )

            # Compute TextGuider loss
            attn_quo_b0 = attn_quo[0] if attn_quo.ndim > 1 else attn_quo
            attn_texts_b0 = [at[0] if at.ndim > 1 else at for at in attn_texts]

            loss = TextGuiderLoss.total_loss(attn_quo_b0, attn_texts_b0)

            if loss.requires_grad:
                loss.backward()

                if latents_grad.grad is not None:
                    grad = latents_grad.grad.detach()
                    updated_latents = original_latents - self.config.alpha * grad

                    print(
                        f"[TextGuider] Step {step}: "
                        f"loss={loss.item():.6f}, "
                        f"grad_norm={grad.norm().item():.6f}"
                    )

                    # Update AMO mask from attention maps
                    if self.amo_sampler is not None:
                        self._amo_mask = self.amo_sampler.compute_overshooting_mask(
                            [at.detach() for at in attn_texts_b0],
                            num_img_tokens=attn_quo_b0.shape[-1],
                            spatial_h=lat_h,
                            spatial_w=lat_w,
                        )

                    return updated_latents
                else:
                    print(f"[TextGuider] Step {step}: No gradient computed (grad is None)")
            else:
                print(f"[TextGuider] Step {step}: Loss has no gradient (loss={loss.item():.6f})")

        except Exception as e:
            print(f"[TextGuider] Step {step}: Guidance failed ({e}), using original latents")

        return original_latents

    def _full_textguider_generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Optional[torch.Generator],
        token_info: Dict,
        t_guide_steps: int,
        **kwargs,
    ) -> Image.Image:
        """Full TextGuider generation with custom denoising loop.

        Used when the pipeline doesn't support callback latents.
        Implements the complete denoising loop with TextGuider guidance
        and AMO overshooting integrated at each step.
        """
        # This is a more invasive approach that replaces the pipeline's
        # built-in denoising loop. We'll use the pipeline's components
        # directly.

        pipe = self.pipe
        transformer = pipe.transformer
        scheduler = pipe.scheduler
        vae = pipe.vae
        tokenizer = pipe.tokenizer

        # Encode text
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
        )

        # Prepare latents
        lat_h = height // 8
        lat_w = width // 8
        num_channels = transformer.config.in_channels if hasattr(transformer, "config") else 128
        shape = (1, (lat_h // 2) * (lat_w // 2), num_channels)
        latents = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)

        # Prepare image IDs
        img_ids = pipe._prepare_latent_image_ids(
            latents.shape[0], lat_h // 2, lat_w // 2, self.device, self.dtype
        )

        # Setup scheduler
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas

        # Attention store
        num_heads = 24
        store = TextGuiderAttentionStore(
            quo_indices=token_info["quo_indices"],
            text_token_indices=token_info["text_token_indices"],
            num_heads=num_heads,
        )
        wrapper = TextGuiderForwardWrapper(
            model=transformer, store=store,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

        print(f"[TextGuider] Starting denoising: {num_inference_steps} steps, "
              f"guidance for first {t_guide_steps} steps")

        for i, t in enumerate(timesteps):
            is_guided = i < t_guide_steps

            if is_guided:
                # TextGuider guided step
                latents_for_grad = latents.detach().clone().requires_grad_(True)

                try:
                    store.clear()
                    attn_quo, attn_texts = wrapper.compute_attention_maps_diffusers(
                        transformer,
                        latents=latents_for_grad,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        timestep=t.unsqueeze(0),
                        img_ids=img_ids,
                        txt_ids=text_ids,
                    )

                    attn_quo_b0 = attn_quo[0]
                    attn_texts_b0 = [at[0] for at in attn_texts]

                    loss = TextGuiderLoss.total_loss(attn_quo_b0, attn_texts_b0)

                    if loss.requires_grad:
                        loss.backward()
                        if latents_for_grad.grad is not None:
                            grad = latents_for_grad.grad.detach()
                            latents = latents.detach() - self.config.alpha * grad
                            print(f"[TextGuider] Step {i}/{num_inference_steps}: "
                                  f"loss={loss.item():.4f}, grad_norm={grad.norm().item():.4f}")

                            # Update AMO mask
                            if self.amo_sampler is not None:
                                self._amo_mask = self.amo_sampler.compute_overshooting_mask(
                                    [at.detach() for at in attn_texts_b0],
                                    num_img_tokens=attn_quo_b0.shape[-1],
                                    spatial_h=lat_h, spatial_w=lat_w,
                                )
                except Exception as e:
                    print(f"[TextGuider] Step {i}: guidance error ({e})")

            # Standard denoising step
            with torch.no_grad():
                timestep_tensor = t.unsqueeze(0).to(self.device)
                noise_pred = transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timestep_tensor / 1000,
                    img_ids=img_ids,
                    txt_ids=text_ids,
                    return_dict=False,
                )[0]

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # AMO overshooting
            if self.config.amo_enabled and self.amo_sampler is not None and self._amo_mask is not None:
                if i + 1 < len(sigmas):
                    sigma_next = float(sigmas[i + 1].item())
                    sigma_curr = float(sigmas[i].item())
                    epsilon = abs(sigma_next - sigma_curr)

                    latents_4d = self._unpack_to_4d(latents, height, width)
                    latents_4d = self.amo_sampler.apply_overshooting(
                        latents_4d, t_next=sigma_next, epsilon=epsilon,
                        mask=self._amo_mask, generator=generator,
                    )
                    latents = self._pack_to_3d(latents_4d)

        # Decode latents to image
        with torch.no_grad():
            latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image)[0]

        return image

    def _baseline_generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: Optional[torch.Generator],
        **kwargs,
    ) -> Image.Image:
        """Run baseline generation without TextGuider."""
        pipe_kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            **kwargs,
        }
        result = self.pipe(**pipe_kwargs)
        return result.images[0]

    def _unpack_to_4d(self, latents: Tensor, height: int, width: int) -> Tensor:
        """Unpack FLUX 3D packed tokens [B, N, D] to 4D [B, C, H, W]."""
        from .latent_utils import unpack_latents
        if latents.ndim == 3:
            return unpack_latents(latents, height=height, width=width)
        return latents

    def _pack_to_3d(self, latents: Tensor) -> Tensor:
        """Pack 4D latents [B, C, H, W] to FLUX 3D tokens [B, N, D]."""
        from .latent_utils import pack_latents
        if latents.ndim == 4:
            return pack_latents(latents)
        return latents

    def _dry_run(self, prompt: str, width: int, height: int) -> Image.Image:
        """Dry-run mode without GPU/model (for testing pipeline logic)."""
        print(f"[TextGuider Dry-Run] Simulating generation for: {prompt[:60]}...")
        img = Image.new("RGB", (width, height), color=(25, 28, 36))
        return img
