"""
Generate Vietnamese Banner/Poster Samples with Base FLUX.2 Klein 4B vs FLUX.2 + TextGuider.
arXiv:2512.09350

Features:
  - Side-by-side comparison images (Base FLUX vs FLUX + TextGuider)
  - Detailed Vietnamese SME/retail prompts (F&B Phở bò, Trà sữa, Bánh mì, Du lịch Áo dài)
  - Qwen3 RoPE patch compatibility for GPU servers
  - GPU memory tracking and performance summary report
  - Supports both full GPU inference and CPU dry-run simulation

Usage:
    # Dry-run on CPU (test script without loading model weights)
    python generate_textguider_samples.py --dry-run

    # Full generation with side-by-side comparison on GPU server
    python generate_textguider_samples.py --compare --seed 42

    # Custom prompt test
    python generate_textguider_samples.py --prompts "Poster khuyến mại 'MUA 1 TẶNG 1' trà sữa" --compare
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont
import torch

# Compatibility patch for Qwen3 used by FLUX.2 Klein on some Triton/CUDA environments
try:
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

    def _custom_rope_forward(self, x, position_ids, **kwargs):
        inv_freq_expanded = self.inv_freq[None, :, None].float()
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded * position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

    qwen3_mod.Qwen3RotaryEmbedding.forward = _custom_rope_forward
except Exception as exc:
    print(f"[Qwen3 RoPE patch] skipped: {exc}")

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.textguider import (
    TextGuiderConfig,
    TextGuiderFluxPipeline,
    TextGuiderLoss,
    TextGuiderTokenParser,
    symmetric_kl_divergence,
)


def _get_font(font_size: int = 24) -> ImageFont.ImageFont:
    """Get a font supporting Vietnamese characters, or default fallback."""
    candidate_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                pass
    return ImageFont.load_default()


def create_side_by_side_comparison(
    img_base: Image.Image,
    img_textguider: Image.Image,
    title: str,
    prompt: str,
) -> Image.Image:
    """Creates a side-by-side comparison image labeled with Base FLUX vs FLUX + TextGuider."""
    w, h = img_base.size
    header_h = 120
    comp_w = w * 2 + 30
    comp_h = h + header_h + 40

    comp_img = Image.new("RGB", (comp_w, comp_h), color=(18, 20, 26))
    draw = ImageDraw.Draw(comp_img)

    font_title = _get_font(font_size=30)
    font_label = _get_font(font_size=24)
    font_sub = _get_font(font_size=16)

    # Draw Header & Prompt
    draw.text((20, 15), f"Tendoo Media AI — {title}", fill=(255, 255, 255), font=font_title)
    short_prompt = (prompt[:120] + "...") if len(prompt) > 120 else prompt
    draw.text((20, 55), f"Prompt: {short_prompt}", fill=(180, 190, 205), font=font_sub)

    # Paste Base Image
    comp_img.paste(img_base, (10, header_h))
    draw.rectangle([10, header_h - 40, 10 + w, header_h], fill=(35, 39, 49))
    draw.text((20, header_h - 35), "❌ Base FLUX.2 Klein (Không TextGuider)", fill=(255, 120, 120), font=font_label)

    # Paste TextGuider Image
    comp_img.paste(img_textguider, (w + 20, header_h))
    draw.rectangle([w + 20, header_h - 40, w + 20 + w, header_h], fill=(20, 45, 60))
    draw.text((w + 30, header_h - 35), "✨ FLUX.2 Klein + TextGuider (arXiv:2512.09350)", fill=(100, 210, 255), font=font_label)

    return comp_img


# Default Vietnamese SME/retail test prompts designed for TextGuider
SAMPLE_PROMPTS = [
    {
        "id": "sample_1_fnb_pho",
        "title": "F&B - Quán Phở Hà Nội",
        "prompt": "Một poster quảng cáo ẩm thực chuyên nghiệp, bát phở bò bốc khói nghi ngút trên bàn gỗ mộc, với dòng chữ nổi bật 'PHỞ BÒ GIA TRUYỀN' và góc dưới có chữ 'GIẢM GIÁ 50%', ánh sáng ấm áp, phong cách studio 4k.",
    },
    {
        "id": "sample_2_cafe_opening",
        "title": "Cafe - Khai Trương Quán Trà Sữa",
        "prompt": "Banner khai trương quán trà sữa hiện đại, ly trà sữa trân châu bắt mắt trên nền pastel hồng cam, chữ nghệ thuật lớn 'KHAI TRƯƠNG' và dòng chữ 'MUA 1 TẶNG 1' nổi bật ở giữa banner.",
    },
    {
        "id": "sample_3_product_banhmi",
        "title": "Sản Phẩm - Bánh Mì Sài Gòn",
        "prompt": "Ảnh chụp sản phẩm bánh mì kẹp thịt Việt Nam giòn rụm, biển hiệu phía sau có ghi rõ chữ 'BÁNH MÌ SÀI GÒN', ánh sáng tự nhiên đẹp mắt, chụp góc cận cảnh macro.",
    },
    {
        "id": "sample_4_travel_aodai",
        "title": "Du Lịch - Áo Dài & Nón Lá",
        "prompt": "Poster quảng bá du lịch Việt Nam, thiếu nữ mặc áo dài trắng duyên dáng đội nón lá bên Hồ Gươm mùa thu, dòng chữ nghệ thuật 'HÀ NỘI MÙA THU' trên nền trời xanh trong trẻo.",
    },
    {
        "id": "sample_5_electronics_sale",
        "title": "Bán Lẻ - Siêu Sale Công Nghệ",
        "prompt": "Poster quảng cáo chương trình ưu đãi thiết bị công nghệ, điện thoại và tai nghe hiện đại với hiệu ứng ánh sáng neon, dòng chữ 3D 'SIÊU SALE 50%' và 'CHÍNH HÃNG 100%'.",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="TextGuider Server Sample Generation & Benchmark")
    parser.add_argument(
        "--model-id", type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
        help="HuggingFace model ID or local weights path"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, default: auto)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="PyTorch dtype")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--steps", type=int, default=50, help="Denoising steps (default: 50 for base)")
    parser.add_argument("--guidance-scale", type=float, default=4.0, help="CFG guidance scale (default: 4.0 for base)")
    parser.add_argument("--alpha", type=float, default=60.0, help="TextGuider guidance step size (default: 60.0)")
    parser.add_argument("--t-guide-ratio", type=float, default=0.25, help="Guidance step fraction (default: 0.25 = 1/4 total steps)")
    parser.add_argument("--amo-c", type=float, default=0.5, help="AMO overshooting hyperparameter c (default: 0.5)")
    parser.add_argument("--no-amo", action="store_true", help="Disable AMO overshooting")
    parser.add_argument("--no-cpu-offload", action="store_true", help="Disable model CPU offload (use only if VRAM >= 24GB)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run on CPU without loading weights")
    parser.add_argument("--compare", action="store_true", help="Generate both Base and TextGuider + side-by-side comparison")
    parser.add_argument("--output-dir", type=str, default="outputs/textguider_server", help="Output directory")
    parser.add_argument("--prompts", type=str, nargs="+", default=None, help="Custom prompt list")
    parser.add_argument("--prompts-file", type=str, default=None, help="Path to JSON/JSONL file containing prompts")
    return parser.parse_args()


def load_prompts(args) -> List[Dict]:
    if args.prompts:
        return [{"id": f"custom_{i+1}", "title": f"Custom Prompt {i+1}", "prompt": p} for i, p in enumerate(args.prompts)]
    if args.prompts_file:
        path = Path(args.prompts_file)
        if path.suffix == ".jsonl":
            results = []
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        item = json.loads(line)
                        results.append({
                            "id": item.get("id", f"case_{idx+1}"),
                            "title": item.get("title", f"Case {idx+1}"),
                            "prompt": item.get("prompt", line.strip()),
                        })
            return results
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
    return SAMPLE_PROMPTS


def main():
    args = parse_args()
    prompts = load_prompts(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    resolved_dtype = dtype_map[args.dtype]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("🚀 FLUX.2 Klein 4B Base + TextGuider Server Runner")
    print(f"  Paper:         TextGuider (arXiv:2512.09350)")
    print(f"  Model ID:      {args.model_id}")
    print(f"  Device:        {device}")
    print(f"  Dtype:         {resolved_dtype}")
    print(f"  Steps:         {args.steps} (Guidance steps: {int(args.steps * args.t_guide_ratio)})")
    print(f"  Guidance CFG:  {args.guidance_scale}")
    print(f"  TextGuider α:  {args.alpha}")
    print(f"  AMO Enabled:   {not args.no_amo} (c = {args.amo_c})")
    print(f"  Resolution:    {args.width}x{args.height}")
    print(f"  Seed:          {args.seed}")
    print(f"  Mode:          {'Dry-run (CPU simulation)' if args.dry_run else 'Full GPU Execution'}")
    print(f"  Compare Mode:  {args.compare}")
    print(f"  Output Dir:    {output_dir}")
    print(f"  Prompt Count:  {len(prompts)}")
    print("=" * 70)

    config = TextGuiderConfig(
        alpha=args.alpha,
        t_guide_ratio=args.t_guide_ratio,
        amo_enabled=not args.no_amo,
        amo_overshoot_c=args.amo_c,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        resolution=args.width,
    )

    if args.dry_run:
        print("[Runner] Running in dry-run mode...")
        pipeline = TextGuiderFluxPipeline(pipe=None, config=config, device="cpu", dtype=torch.float32)
    else:
        print(f"[Runner] Loading FLUX.2 Klein pipeline ({args.model_id})...")
        pipeline = TextGuiderFluxPipeline.from_pretrained(
            model_id=args.model_id,
            config=config,
            device=device,
            torch_dtype=resolved_dtype,
            enable_cpu_offload=not args.no_cpu_offload,
        )

    summary_records = []

    for i, item in enumerate(prompts):
        pid = item.get("id", f"sample_{i+1}")
        title = item.get("title", f"Sample {i+1}")
        prompt = item["prompt"]

        print(f"\n[{i+1}/{len(prompts)}] 📌 {title}")
        print(f"    Prompt: {prompt}")

        t0 = time.time()
        record = {
            "id": pid,
            "title": title,
            "prompt": prompt,
            "seed": args.seed,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "alpha": args.alpha,
            "t_guide_ratio": args.t_guide_ratio,
        }

        # 1. Base generation (if compare mode enabled)
        img_base = None
        if args.compare:
            print("    ▶ Generating Base FLUX.2 Klein...")
            t_base_start = time.time()
            img_base = pipeline.generate(
                prompt=prompt,
                width=args.width,
                height=args.height,
                seed=args.seed,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                use_textguider=False,
            )
            base_time = time.time() - t_base_start
            record["base_time_sec"] = round(base_time, 2)
            base_path = output_dir / f"{pid}_base.png"
            img_base.save(base_path)
            print(f"    ✓ Base saved: {base_path} ({base_time:.1f}s)")

        # 2. TextGuider generation
        print("    ▶ Generating FLUX.2 Klein + TextGuider...")
        t_tg_start = time.time()
        img_tg = pipeline.generate(
            prompt=prompt,
            width=args.width,
            height=args.height,
            seed=args.seed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            use_textguider=True,
        )
        tg_time = time.time() - t_tg_start
        record["textguider_time_sec"] = round(tg_time, 2)
        tg_path = output_dir / f"{pid}_textguider.png"
        img_tg.save(tg_path)
        print(f"    ✓ TextGuider saved: {tg_path} ({tg_time:.1f}s)")

        # 3. Create side-by-side comparison if compare mode is on
        if args.compare and img_base is not None:
            comp_img = create_side_by_side_comparison(img_base, img_tg, title, prompt)
            comp_path = output_dir / f"{pid}_comparison.png"
            comp_img.save(comp_path)
            print(f"    ✓ Side-by-side comparison saved: {comp_path}")

        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            record["max_vram_gb"] = round(vram_gb, 2)
            print(f"    ⚡ Peak VRAM: {vram_gb:.2f} GB")

        summary_records.append(record)

    # Save summary json
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": args.model_id,
            "device": device,
            "dtype": str(resolved_dtype),
            "config": {
                "alpha": args.alpha,
                "t_guide_ratio": args.t_guide_ratio,
                "amo_enabled": not args.no_amo,
                "amo_overshoot_c": args.amo_c,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
            },
            "results": summary_records,
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"✅ Hoàn thành sinh mẫu TextGuider!")
    print(f"📁 Kết quả lưu tại: {output_dir.resolve()}")
    print(f"📄 File tổng kết:   {summary_file.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
