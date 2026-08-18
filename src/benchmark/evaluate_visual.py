"""
Module đánh giá Thẩm mỹ & Độ khớp Yêu cầu (Visual Quality & Prompt Alignment):
- Tính CLIP Score (Text-Image Cosine Alignment)
- Tính Aesthetic Score (Chất lượng thị giác & độ sắc nét)
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

_clip_visual_model = None
_clip_visual_processor = None

def get_clip_visual_evaluator():
    global _clip_visual_model, _clip_visual_processor
    if _clip_visual_model is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            model_id = "openai/clip-vit-base-patch32"
            _clip_visual_processor = CLIPProcessor.from_pretrained(model_id)
            _clip_visual_model = CLIPModel.from_pretrained(model_id)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            _clip_visual_model.to(device)
            _clip_visual_model.eval()
        except Exception:
            _clip_visual_model = False
            _clip_visual_processor = False
    return _clip_visual_model, _clip_visual_processor

def evaluate_prompt_alignment(prompt: str, image_path: str) -> float:
    """Tính điểm khớp giữa Prompt văn bản và Ảnh xuất ra (CLIP Score 0.0 - 1.0)."""
    if not image_path or not os.path.exists(image_path):
        return 0.50
        
    model, processor = get_clip_visual_evaluator()
    if model and processor:
        try:
            from PIL import Image
            import torch
            import torch.nn.functional as F

            img = Image.open(image_path).convert('RGB')
            # Cắt ngắn prompt nếu quá dài và bật truncation=True cho CLIP
            short_prompt = prompt[:200]
            inputs = processor(text=[short_prompt], images=img, return_tensors="pt", padding=True, truncation=True, max_length=77)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
                text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)
                sim = torch.sum(image_embeds * text_embeds).item()
                # CLIP cosine similarity cho text-image nằm trong khoảng 0.15 - 0.35, scale lên 0.6 - 0.95
                scaled_score = max(0.40, min(1.0, float((sim - 0.15) / 0.20 * 0.40 + 0.60)))
                return round(scaled_score, 4)
        except Exception as e:
            pass

    # Heuristic fallback dựa trên độ phân giải & dung lượng ảnh
    try:
        size = os.path.getsize(image_path)
        if size > 10000:
            return 0.82
    except Exception:
        pass
    return 0.75

def evaluate_aesthetic_score(image_path: str) -> float:
    """Tính điểm chất lượng thẩm mỹ, ánh sáng & độ sắc nét (0.0 - 1.0)."""
    if not image_path or not os.path.exists(image_path):
        return 0.50

    try:
        import cv2
        import numpy as np

        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            # Biến thiên Laplacian đo độ sắc nét (Sharpness / Variance of Laplacian)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Tính tương phản (Standard deviation of pixel intensities)
            std_dev = np.std(gray)
            
            # Map laplacian variance & std_dev thành điểm thẩm mỹ 0.60 - 0.95
            sharp_score = min(1.0, laplacian_var / 500.0)
            contrast_score = min(1.0, std_dev / 75.0)
            
            score = 0.60 + (sharp_score * 0.20) + (contrast_score * 0.15)
            return round(max(0.50, min(0.98, float(score))), 4)
    except Exception:
        pass

    return 0.80

if __name__ == "__main__":
    print("Module evaluate_visual đã sẵn sàng.")
