"""Runtime probes for the FLUX.2 Klein assumptions in ARCHITECTURE_NOTES.md.

Run on the GPU host, after installing the pinned environment:

    python tools/verify_flux2_architecture.py --model black-forest-labs/FLUX.2-klein-base-4B

The script intentionally loads the real pipeline and emits JSON.  It does not
generate an image, so it is suitable as a cheap first checkpoint/API audit.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from textguider.latent_utils import pack_latents, unpack_latents  # noqa: E402
from textguider.textguider_attention import _is_double_stream_attention  # noqa: E402


def shape_of(value: Any):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (tuple, list)):
        return [shape_of(item) for item in value]
    return type(value).__name__


def source_has(source: str, *needles: str) -> bool:
    return all(needle in source for needle in needles)


def install_qwen3_rope_patch() -> str:
    """Avoid the Triton-only Qwen3 RoPE path on restricted GPU servers.

    Some hosted images cannot compile Triton's ``cuda_utils`` because the
    Python development header is unavailable. This mathematically equivalent
    implementation uses ordinary PyTorch broadcasting instead.
    """
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

        if getattr(qwen3_mod.Qwen3RotaryEmbedding, "_tendoo_rope_patch", False):
            return "already-installed"

        def custom_rope_forward(self, x, position_ids, **kwargs):
            inv_freq_expanded = self.inv_freq[None, :, None].float()
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded * position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

        qwen3_mod.Qwen3RotaryEmbedding.forward = custom_rope_forward
        qwen3_mod.Qwen3RotaryEmbedding._tendoo_rope_patch = True
        return "installed"
    except Exception as exc:
        return f"failed: {exc!r}"


def flux2_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Match Diffusers' Flux2 dynamic-shifting schedule calculation."""
    try:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        return float(compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps))
    except (ImportError, AttributeError):
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--prompt", default='A poster with the words "MUA 1 TANG 1"')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this probe on the GPU host.")

    from diffusers import Flux2KleinPipeline

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    rope_patch_status = install_qwen3_rope_patch()
    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.to(args.device)
    transformer = pipe.transformer
    config = transformer.config

    report: dict[str, Any] = {
        "model": args.model,
        "device": args.device,
        "dtype": str(dtype),
        "qwen3_rope_patch": rope_patch_status,
    }
    report["classes"] = {
        "pipeline": type(pipe).__name__,
        "transformer": type(transformer).__name__,
        "text_encoder": type(pipe.text_encoder).__name__,
        "tokenizer": type(pipe.tokenizer).__name__,
        "scheduler": type(pipe.scheduler).__name__,
    }
    report["transformer_config"] = {
        key: getattr(config, key, None)
        for key in ("in_channels", "out_channels", "num_layers", "num_single_layers",
                    "num_attention_heads", "attention_head_dim", "joint_attention_dim",
                    "guidance_embeds")
    }
    report["forward_signature"] = str(inspect.signature(transformer.forward))
    report["encode_prompt_signature"] = str(inspect.signature(pipe.encode_prompt))

    modules = [(name, module) for name, module in transformer.named_modules()
               if _is_double_stream_attention(module)]
    report["double_stream"] = {
        "count": len(modules),
        "names": [name for name, _ in modules],
        "config_num_layers": getattr(config, "num_layers", None),
        "matches_config": len(modules) == getattr(config, "num_layers", -1),
    }
    report["single_stream"] = {
        "config_num_single_layers": getattr(config, "num_single_layers", None),
        "module_name_hints": [name for name, _ in transformer.named_modules()
                               if "single" in name.lower()][:32],
    }

    with torch.no_grad():
        encoded = pipe.encode_prompt(prompt=args.prompt, device=args.device)
    report["encode_prompt_output"] = shape_of(encoded)
    report["encode_prompt_is_two_tuple"] = isinstance(encoded, tuple) and len(encoded) == 2
    prompt_embeds, text_ids = encoded[0], encoded[1]
    report["prompt_embed_seq_len"] = int(prompt_embeds.shape[1])
    report["text_ids_shape"] = list(text_ids.shape) if isinstance(text_ids, torch.Tensor) else None

    raw = pipe.tokenizer(args.prompt, padding=False, truncation=True,
                         max_length=getattr(pipe.tokenizer, "model_max_length", 512),
                         return_attention_mask=True)
    report["raw_tokenizer"] = {"input_ids": len(raw["input_ids"]),
                               "attention_tokens": int(sum(raw["attention_mask"]))}
    try:
        chat = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        report["chat_template_token_count"] = len(chat)
    except Exception as exc:
        report["chat_template_token_count"] = None
        report["chat_template_error"] = repr(exc)

    report["timestep_probe"] = {
        "scheduler_timesteps_before_set": list(getattr(pipe.scheduler, "timesteps", []))[:3],
        "prepare_rule_expected": "timestep / 1000 (matches Flux2KleinPipeline source)",
        "transformer_call_should_use_guidance": None,
    }
    # Flux2's scheduler uses dynamic shifting and therefore requires mu.
    # At the probe's default 1024x1024 resolution the packed image sequence
    # is (1024 // 16) * (1024 // 16) = 4096 tokens.
    probe_steps = 2
    probe_image_seq_len = (1024 // 16) * (1024 // 16)
    probe_mu = flux2_empirical_mu(probe_image_seq_len, probe_steps)
    pipe.scheduler.set_timesteps(probe_steps, device=args.device, mu=probe_mu)
    report["scheduler_timesteps_after_set"] = [float(x) for x in pipe.scheduler.timesteps]
    report["scheduler_sigmas"] = [float(x) for x in pipe.scheduler.sigmas]
    report["scheduler_mu"] = probe_mu

    # Prove the fallback permutation is reversible; the live pipeline methods
    # are reported separately because their signatures vary by Diffusers version.
    x = torch.randn(1, 32, 8, 10, device=args.device, dtype=dtype)
    packed = pack_latents(x)
    restored = unpack_latents(packed, height=64, width=80)
    report["fallback_pack_roundtrip_max_abs"] = float((x - restored).abs().max().cpu())
    report["pipeline_pack_methods"] = {
        "pack": str(inspect.signature(pipe._pack_latents)) if hasattr(pipe, "_pack_latents") else None,
        "unpack": str(inspect.signature(pipe._unpack_latents)) if hasattr(pipe, "_unpack_latents") else None,
    }

    try:
        source = inspect.getsource(type(pipe).__call__)
    except (OSError, TypeError):
        source = ""
    report["official_call_source_checks"] = {
        "divides_timestep_by_1000": "timestep / 1000" in source,
        "passes_guidance_none": source_has(source, "guidance=None"),
        "contains_cfg_scale": "guidance_scale" in source,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    failures = []
    if report["classes"]["pipeline"] != "Flux2KleinPipeline": failures.append("pipeline class")
    if not report["encode_prompt_is_two_tuple"]: failures.append("encode_prompt is not 2-tuple")
    if not report["double_stream"]["matches_config"]: failures.append("double block count")
    if report["fallback_pack_roundtrip_max_abs"] != 0.0: failures.append("pack roundtrip")
    if not report["official_call_source_checks"]["divides_timestep_by_1000"]: failures.append("timestep source")
    if failures:
        print("FAIL: " + ", ".join(failures), file=sys.stderr)
        return 2
    print("PASS: architecture probes completed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
