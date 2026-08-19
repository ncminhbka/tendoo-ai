"""
Generate Vietnamese Banner/Poster Samples with Base FLUX.2 vs FLUX.2 + FreeText.
Generates both outputs and a side-by-side comparison image for easy manual inspection.
"""

import os
import sys
import argparse
import time
from PIL import Image, ImageDraw, ImageFont
import torch

# Compatibility patch for Qwen3 used by FLUX.2 Klein on some Triton/CUDA
# combinations.  Keep this before loading the Diffusers pipeline so the text
# encoder uses the patched RoPE implementation from its first forward pass.
try:
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod

    def custom_rope_forward(self, x, position_ids, **kwargs):
        # inv_freq: (dim / 2,) -> (1, dim / 2, 1)
        # position_ids: (batch, seq_len) -> (batch, 1, seq_len)
        inv_freq_expanded = self.inv_freq[None, :, None].float()
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded * position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Match query/key/value dtype (normally bfloat16 on the GPU server).
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

    qwen3_mod.Qwen3RotaryEmbedding.forward = custom_rope_forward
except Exception as exc:
    print(f"[Qwen3 RoPE patch] skipped: {exc}")

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from freetext import FreeTextFluxPipeline, FreeTextConfig, get_vietnamese_font

SAMPLE_PROMPTS = [
    {
        "id": "sample_1_fnb",
        "title": "F&B - Phở Bò",
        "prompt": "Một poster quảng cáo ẩm thực sang trọng, món phở Việt Nam thơm ngon bốc khói nghi ngút, trên banner có dòng chữ nổi bật 'PHỞ BÒ GIA TRUYỀN', giảm giá 'ƯU ĐÃI 30%', phong cách chụp ảnh studio chuyên nghiệp 4k.",
        "texts": ["PHỞ BÒ GIA TRUYỀN", "ƯU ĐÃI 30%"],
    },
    {
        "id": "sample_2_cafe",
        "title": "Cafe - Cà Phê Sữa Đá",
        "prompt": "Poster quảng cáo quán cà phê hiện đại, ly 'CÀ PHÊ SỮA ĐÁ' đậm đà với đá viên và hạt cà phê rơi xung quanh, ánh sáng ấm áp, typography chữ 'ĐẬM ĐÀ HƯƠNG VỊ VIỆT'.",
        "texts": ["CÀ PHÊ SỮA ĐÁ", "ĐẬM ĐÀ HƯƠNG VỊ VIỆT"],
    },
    {
        "id": "sample_3_opening",
        "title": "Banner Khai Trương",
        "prompt": "Banner sự kiện khai trương cửa hàng rực rỡ với bóng bay và pháo hoa vàng kim, chữ nghệ thuật lớn 'KHAI TRƯƠNG HỒNG PHÁT', kèm dòng chữ 'GIẢM GIÁ 50%' toàn bộ sản phẩm.",
        "texts": ["KHAI TRƯƠNG HỒNG PHÁT", "GIẢM GIÁ 50%"],
    },
    {
        "id": "sample_4_travel",
        "title": "Du Lịch - Áo Dài",
        "prompt": "Poster du lịch Việt Nam tuyệt đẹp, cô gái mặc áo dài trắng duyên dáng đội nón lá bên Hồ Gươm mùa thu, dòng chữ nghệ thuật 'HÀ NỘI MÙA THU' và 'CHÀO ĐÓN DU KHÁCH'.",
        "texts": ["HÀ NỘI MÙA THU", "CHÀO ĐÓN DU KHÁCH"],
    },
]


def create_side_by_side_comparison(
    img_base: Image.Image,
    img_freetext: Image.Image,
    title: str,
    prompt: str,
) -> Image.Image:
    """
    Creates a side-by-side comparison image labeled with Base FLUX vs FLUX + FreeText.
    """
    w, h = img_base.size
    header_h = 120
    comp_w = w * 2 + 30
    comp_h = h + header_h + 40

    comp_img = Image.new("RGB", (comp_w, comp_h), color=(18, 20, 26))
    draw = ImageDraw.Draw(comp_img)

    font_title = get_vietnamese_font(font_size=32)
    font_label = get_vietnamese_font(font_size=28)
    font_sub = get_vietnamese_font(font_size=18)

    # Draw Title and Prompt
    draw.text((20, 15), f"Tendoo Media AI — {title}", fill=(255, 255, 255), font=font_title)
    short_prompt = (prompt[:110] + "...") if len(prompt) > 110 else prompt
    draw.text((20, 55), f"Prompt: {short_prompt}", fill=(180, 190, 205), font=font_sub)

    # Paste Base Image
    comp_img.paste(img_base, (10, header_h))
    draw.rectangle([10, header_h - 40, 10 + w, header_h], fill=(35, 39, 49))
    draw.text((20, header_h - 35), "❌ Base FLUX.2 (Không FreeText)", fill=(255, 120, 120), font=font_label)

    # Paste FreeText Image
    comp_img.paste(img_freetext, (w + 20, header_h))
    draw.rectangle([w + 20, header_h - 40, w + 20 + w, header_h], fill=(24, 48, 36))
    draw.text((w + 30, header_h - 35), "✨ FLUX.2 + FreeText (SGMI Active)", fill=(100, 255, 150), font=font_label)

    return comp_img


def main():
    parser = argparse.ArgumentParser(description="Generate Vietnamese Banner Samples: Base FLUX.2 vs FLUX.2 + FreeText")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.2-klein-base-4B", help="Model checkpoint path or HF ID")
    parser.add_argument("--output-dir", type=str, default="outputs/freetext_test", help="Directory to save output images")
    parser.add_argument("--steps", type=int, default=50, help="Inference sampling steps (50 for base, 4 for distilled)")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of sample prompts to run")
    parser.add_argument("--dry-run", action="store_true", help="Run in CPU dry-run simulation mode")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 Tendoo Media AI — FreeText (arXiv:2601.00535) Sample Generator")
    print(f"   Model: {args.model_id}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   Steps: {args.steps} | Guidance: {args.guidance} | Seed: {args.seed}")
    print("=" * 70)

    # Initialize Pipeline
    if args.dry_run:
        print("[Mode] Running in CPU Dry-Run mode...")
        pipe = FreeTextFluxPipeline(pipe=None)
    else:
        try:
            pipe = FreeTextFluxPipeline.from_pretrained(
                model_id=args.model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                enable_cpu_offload=True,
            )
        except Exception as e:
            print(f"[Warning] Could not load model ({e}). Full details:")
            import traceback
            traceback.print_exc()
            print("Falling back to Dry-Run mode.")
            pipe = FreeTextFluxPipeline(pipe=None)

    prompts_to_run = SAMPLE_PROMPTS[:args.num_samples] if args.num_samples else SAMPLE_PROMPTS

    for i, item in enumerate(prompts_to_run, 1):
        sample_id = item["id"]
        title = item["title"]
        prompt = item["prompt"]
        texts = item.get("texts")

        print(f"\n[{i}/{len(prompts_to_run)}] Processing: {title}")
        print(f"    Prompt: {prompt}")
        print(f"    Targets: {texts}")

        # 1. Generate with Base FLUX (FreeText Disabled)
        print("  -> Generating [1/2] Base FLUX.2...")
        t0 = time.time()
        img_base = pipe.generate(
            prompt=prompt,
            target_texts=texts,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            use_freetext=False,
        )
        base_time = time.time() - t0
        base_path = os.path.join(args.output_dir, f"{sample_id}_base.png")
        img_base.save(base_path)

        # 2. Generate with FLUX + FreeText (SGMI Enabled)
        print("  -> Generating [2/2] FLUX.2 + FreeText...")
        t1 = time.time()
        img_freetext = pipe.generate(
            prompt=prompt,
            target_texts=texts,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            use_freetext=True,
        )
        freetext_time = time.time() - t1
        freetext_path = os.path.join(args.output_dir, f"{sample_id}_freetext.png")
        img_freetext.save(freetext_path)

        # 3. Create Side-by-Side Comparison
        comp_img = create_side_by_side_comparison(img_base, img_freetext, title=title, prompt=prompt)
        comp_path = os.path.join(args.output_dir, f"{sample_id}_comparison.png")
        comp_img.save(comp_path)

        print(f"  [OK] Saved: {base_path} ({base_time:.2f}s)")
        print(f"  [OK] Saved: {freetext_path} ({freetext_time:.2f}s)")
        print(f"  [OK] Comparison: {comp_path}")

    print("\n" + "=" * 70)
    print(f"🎉 Hoàn thành sinh mẫu! Toàn bộ ảnh kết quả và ảnh so sánh đã lưu tại: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
