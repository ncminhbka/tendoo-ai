"""
TextGuider Pipeline for FLUX.2 Klein 4B Base.
arXiv:2512.09350

Tích hợp TextGuider latent guidance với AMO Sampler cho FLUX.2 Klein 4B
Base, bám theo hành vi thật của model Base đã xác nhận từ repo chính thức
(black-forest-labs/flux2) — xem ARCHITECTURE_NOTES.md.

Thay đổi cốt lõi so với bản trước:
  1. `encode_prompt` được coi là trả về 2-tuple (embeds, txt_ids) — KHÔNG
     có pooled_projections (Flux2Transformer2DModel không nhận tham số
     này). Bản cũ có 2 chỗ giả định khác nhau (2-tuple ở callback path,
     3-tuple ở path native-loop) — nay thống nhất một chỗ duy nhất.
  2. Model Base dùng classifier-free guidance THẬT (2 lượt forward: có
     điều kiện + không điều kiện, trộn theo guidance_scale) — không phải
     một lượt với guidance nhúng sẵn như model đã distill. Thêm hẳn
     encode uncond prompt + công thức trộn CFG.
  3. Timestep được chuẩn hoá qua một hàm DUY NHẤT (`_prepare_timestep`),
     dùng lại y hệt ở mọi lệnh gọi transformer trong cùng một bước — bản
     cũ có 2 kiểu scale khác nhau ngay trong cùng một step.
  4. Pack/unpack latent ưu tiên gọi hàm gốc của `pipe` (nếu có) thay vì
     luôn dùng bản tự viết.
  5. Token index được verify khớp với encoder_hidden_states thật trước
     khi dùng cho guidance (xem TextGuiderTokenParser.verify_alignment).
  6. strict_mode: lỗi hook/alignment được raise rõ ràng thay vì nuốt hết
     bằng except-print-fallback (vẫn có chế độ non-strict cho production
     một khi đã xác nhận pipeline chạy đúng).
"""

from __future__ import annotations

import sys
from typing import Dict, Optional

import torch
from torch import Tensor
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .textguider import (
    TextGuiderConfig,
    TextGuiderLoss,
    TextGuiderTokenParser,
    TokenAlignmentError,
    AMOSampler,
)
from .textguider_attention import (
    TextGuiderAttentionStore,
    TextGuiderForwardWrapper,
    AttentionCaptureError,
)
from .latent_utils import get_pack_unpack_fns


class TextGuiderFluxPipeline:
    """FLUX.2 Klein 4B Base pipeline với TextGuider text rendering enhancement."""

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

        self.amo_sampler = AMOSampler(self.config) if self.config.amo_enabled else None
        self._amo_mask: Optional[Tensor] = None

        self._pack_fn, self._unpack_fn = get_pack_unpack_fns(pipe)

    @property
    def is_live(self) -> bool:
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
        device_map: Optional[str] = None,
        **kwargs,
    ):
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
            load_kwargs = dict(kwargs)
            if device_map is not None:
                load_kwargs["device_map"] = device_map
            pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=resolved_dtype, **load_kwargs)
            if device_map is not None:
                print(f"[TextGuider] Device map: {device_map}; keeping pipeline sharded")
            elif enable_cpu_offload and device.startswith("cuda") and hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            elif hasattr(pipe, "to"):
                pipe.to(device)

        return cls(pipe=pipe, config=config, device=device, dtype=resolved_dtype)

    # ------------------------------------------------------------------
    # Timestep — MỘT nguồn duy nhất cho toàn bộ pipeline.
    # ------------------------------------------------------------------

    def _prepare_timestep(self, t: Tensor, latents: Tensor) -> Tensor:
        """Chuẩn hoá timestep theo đúng 1 quy ước, dùng lại ở MỌI lệnh gọi
        transformer trong cùng một bước denoise.

        Quy ước /1000 khớp với cách các pipeline Flux họ nhà Diffusers vẫn
        làm (scheduler.timesteps ở thang huấn luyện ~[0,1000], transformer
        nhận timestep đã chuẩn hoá ~[0,1]). Nếu bạn xác nhận qua log thật
        rằng pipeline của mình dùng quy ước khác, chỉ cần sửa DUY NHẤT ở
        đây — mọi nơi gọi transformer sẽ tự động nhất quán theo.
        """
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.tensor([float(t)], device=latents.device, dtype=latents.dtype)
        else:
            t_tensor = t.unsqueeze(0) if t.ndim == 0 else t.flatten()
            t_tensor = t_tensor.to(device=latents.device, dtype=latents.dtype)
        if latents.shape[0] > 1 and t_tensor.shape[0] == 1:
            t_tensor = t_tensor.expand(latents.shape[0])
        return t_tensor / 1000.0

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

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
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale

        if generator is None and seed is not None:
            gen_device = self.device if self.device != "cpu" and torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=gen_device).manual_seed(seed)

        if self.pipe is None:
            return self._dry_run(prompt, width, height)

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
            return self._baseline_generate(
                prompt, width, height, num_inference_steps, guidance_scale, generator, **kwargs
            )

        t_guide_steps = max(1, int(num_inference_steps * self.config.t_guide_ratio))
        print(
            f"[TextGuider] Config: alpha={self.config.alpha}, "
            f"t_guide={t_guide_steps}/{num_inference_steps} steps, "
            f"AMO={'on' if self.config.amo_enabled else 'off'} "
            f"(c={self.config.amo_overshoot_c}), CFG={self.config.use_cfg}"
        )

        # Đường denoising thủ công (đầy đủ, có CFG thật + TextGuider + AMO).
        # Đây là đường được khuyến nghị cho model Base — xem
        # ARCHITECTURE_NOTES.md mục 4 vì sao callback đơn giản không đủ.
        return self._full_textguider_generate(
            prompt, width, height, num_inference_steps, guidance_scale,
            generator, token_info, t_guide_steps, **kwargs
        )

    def _encode(self, pipe, prompt: str):
        """Gọi pipe.encode_prompt và chuẩn hoá về đúng 2-tuple (embeds, ids).

        Flux2Transformer2DModel không có pooled_projections (xác nhận —
        xem ARCHITECTURE_NOTES.md mục 2), nên bất kể encode_prompt trả về
        bao nhiêu giá trị, ta chỉ lấy đúng embeds + txt_ids.
        """
        res = pipe.encode_prompt(prompt=prompt)
        if isinstance(res, tuple):
            prompt_embeds = res[0]
            txt_ids = res[-1] if len(res) > 1 else None
        else:
            prompt_embeds = res
            txt_ids = None
        return prompt_embeds, txt_ids

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
        """Denoising loop đầy đủ: CFG thật (nếu use_cfg) + TextGuider + AMO."""
        pipe = self.pipe
        transformer = pipe.transformer
        scheduler = pipe.scheduler
        vae = pipe.vae

        # TextGuider differentiates with respect to the latent, never with
        # respect to model weights. Freezing the transformer parameters is
        # essential on 24 GB GPUs: otherwise autograd also allocates gradient
        # buffers for the 4B model during the attention-guidance backward.
        transformer_was_training = transformer.training
        transformer.eval()
        for parameter in transformer.parameters():
            parameter.requires_grad_(False)

        prompt_embeds, txt_ids = self._encode(pipe, prompt)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype).detach()
        if isinstance(txt_ids, torch.Tensor):
            txt_ids = txt_ids.to(device=self.device).detach()

        uncond_embeds, uncond_ids = None, None
        if self.config.use_cfg:
            uncond_embeds, uncond_ids = self._encode(pipe, self.config.negative_prompt)
            uncond_embeds = uncond_embeds.to(device=self.device, dtype=self.dtype).detach()
            if isinstance(uncond_ids, torch.Tensor):
                uncond_ids = uncond_ids.to(device=self.device).detach()

        # Verify token alignment TRƯỚC khi dùng cho guidance — xem
        # ARCHITECTURE_NOTES.md mục 3.
        try:
            TextGuiderTokenParser.verify_alignment(
                token_info, prompt_embeds, strict=self.config.strict_mode
            )
        except TokenAlignmentError as e:
            if self.config.strict_mode:
                raise
            print(f"[TextGuider] {e}\n[TextGuider] Tiếp tục vì strict_mode=False, "
                  f"nhưng guidance có thể sai vị trí.")

        lat_h = height // 8
        lat_w = width // 8
        num_channels = transformer.config.in_channels if hasattr(transformer, "config") else 128
        shape = (1, (lat_h // 2) * (lat_w // 2), num_channels)
        latents = torch.randn(shape, device=self.device, dtype=self.dtype, generator=generator)

        # Diffusers renamed the Flux2 helper from _prepare_latent_image_ids
        # to _prepare_latent_ids. The latter consumes the unpacked 4D latent
        # shape, while our denoising loop already stores packed 3D latents.
        if hasattr(pipe, "_prepare_latent_ids"):
            latent_shape = (
                latents.shape[0], num_channels // 4, lat_h // 2, lat_w // 2
            )
            latent_shape_probe = torch.empty(
                latent_shape, device=self.device, dtype=self.dtype
            )
            img_ids = pipe._prepare_latent_ids(latent_shape_probe).to(self.device)
        elif hasattr(pipe, "_prepare_latent_image_ids"):
            img_ids = pipe._prepare_latent_image_ids(
                latents.shape[0], lat_h // 2, lat_w // 2, self.device, self.dtype
            )
        else:
            raise AttributeError(
                "Flux2 pipeline exposes neither _prepare_latent_ids nor "
                "_prepare_latent_image_ids"
            )

        if getattr(scheduler.config, "use_dynamic_shifting", False):
            try:
                from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
                mu = compute_empirical_mu(
                    image_seq_len=latents.shape[1], num_steps=num_inference_steps
                )
            except (ImportError, AttributeError):
                # Same constants as Diffusers Flux2 for older package builds.
                a1, b1 = 8.73809524e-05, 1.89833333
                a2, b2 = 0.00016927, 0.45666666
                n = latents.shape[1]
                if n > 4300:
                    mu = a2 * n + b2
                else:
                    m200 = a2 * n + b2
                    m10 = a1 * n + b1
                    slope = (m200 - m10) / 190.0
                    mu = slope * num_inference_steps + (m200 - 200.0 * slope)
            scheduler.set_timesteps(num_inference_steps, device=self.device, mu=mu)
        else:
            scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas

        store = TextGuiderAttentionStore(
            quo_indices=token_info["quo_indices"],
            text_token_indices=token_info["text_token_indices"],
        )
        wrapper = TextGuiderForwardWrapper(
            model=transformer, store=store,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            strict_mode=self.config.strict_mode,
        )

        num_layers = getattr(transformer.config, "num_layers", None)
        num_single_layers = getattr(transformer.config, "num_single_layers", None)
        print(f"[TextGuider] transformer.config: num_layers={num_layers}, "
              f"num_single_layers={num_single_layers} (đọc động, không hardcode)")

        print(f"[TextGuider] Starting denoising: {num_inference_steps} steps, "
              f"guidance for first {t_guide_steps} steps, CFG={self.config.use_cfg}")

        for i, t in enumerate(timesteps):
            is_guided = i < t_guide_steps
            t_step = self._prepare_timestep(t, latents).detach()
            # The official Flux2Klein CFG path passes guidance=None to the
            # transformer. CFG scale is applied after conditional and
            # unconditional predictions are produced; it is not the
            # transformer's optional guidance embedding.
            guidance_tensor = None

            if is_guided:
                latents_for_grad = latents.detach().clone().requires_grad_(True)
                try:
                    store.clear()
                    attn_quo, attn_texts = wrapper.compute_attention_maps_diffusers(
                        transformer,
                        latents=latents_for_grad,
                        encoder_hidden_states=prompt_embeds,
                        timestep=t_step,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance_tensor,
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

                            if self.amo_sampler is not None:
                                self._amo_mask = self.amo_sampler.compute_overshooting_mask(
                                    [at.detach() for at in attn_texts_b0],
                                    num_img_tokens=attn_quo_b0.shape[-1],
                                    spatial_h=lat_h, spatial_w=lat_w,
                                )
                except (AttentionCaptureError, TokenAlignmentError):
                    if self.config.strict_mode:
                        raise
                    print(f"[TextGuider] Step {i}: guidance thất bại (strict_mode=False), bỏ qua.")
                except Exception as e:
                    if self.config.strict_mode:
                        raise
                    print(f"[TextGuider] Step {i}: guidance error ({e})")

            # Denoise thật — CFG nếu use_cfg, ngược lại một lượt duy nhất.
            with torch.no_grad():
                if self.config.use_cfg and uncond_embeds is not None:
                    noise_pred_cond = transformer(
                        hidden_states=latents,
                        encoder_hidden_states=prompt_embeds,
                        timestep=t_step,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond = transformer(
                        hidden_states=latents,
                        encoder_hidden_states=uncond_embeds,
                        timestep=t_step,
                        img_ids=img_ids,
                        txt_ids=uncond_ids if uncond_ids is not None else txt_ids,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    # Công thức CFG kinh điển, khớp denoise_cfg() của repo
                    # gốc cho model Base (guidance mặc định 4.0).
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )
                else:
                    noise_pred = transformer(
                        hidden_states=latents,
                        encoder_hidden_states=prompt_embeds,
                        timestep=t_step,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]

                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if self.config.amo_enabled and self.amo_sampler is not None and self._amo_mask is not None:
                if i + 1 < len(sigmas):
                    sigma_next = float(sigmas[i + 1].item())
                    sigma_curr = float(sigmas[i].item())
                    epsilon = abs(sigma_next - sigma_curr)

                    latents_4d = self._unpack_fn(latents, height, width)
                    latents_4d = self.amo_sampler.apply_overshooting(
                        latents_4d, t_next=sigma_next, epsilon=epsilon,
                        mask=self._amo_mask, generator=generator,
                    )
                    latents = self._pack_fn(latents_4d, height, width)

        with torch.no_grad():
            latents_decoded = self._unpack_fn(latents, height, width)
            # Diffusers may expose the VAE config as a FrozenDict without
            # attribute accessors on the sharded pipeline.
            scaling_factor = (
                vae.config.get("scaling_factor", 1.0)
                if hasattr(vae.config, "get")
                else getattr(vae.config, "scaling_factor", 1.0)
            )
            shift_factor = (
                vae.config.get("shift_factor", 0.0)
                if hasattr(vae.config, "get")
                else getattr(vae.config, "shift_factor", 0.0)
            )
            latents_decoded = (latents_decoded / scaling_factor) + shift_factor
            image = vae.decode(latents_decoded, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image)[0]

        if transformer_was_training:
            transformer.train()

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
        """Baseline: giao hẳn cho pipeline gốc — pipeline Diffusers tự lo
        CFG/guidance đúng cách cho model, không cần ta can thiệp."""
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

    def _dry_run(self, prompt: str, width: int, height: int) -> Image.Image:
        print(f"[TextGuider Dry-Run] Simulating generation for: {prompt[:60]}...")
        img = Image.new("RGB", (width, height), color=(25, 28, 36))
        return img
