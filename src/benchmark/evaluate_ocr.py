"""
Module đánh giá Text Accuracy cho Tiếng Việt:
- Chuẩn hóa Unicode NFC
- Tính Character Error Rate (CER), Word Error Rate (WER) bằng Substring Matching
- Tính Exact Match Ratio và Normalized Edit Distance (NED)
- Tích hợp VietOCR + Text Detection (Cropping) & EasyOCR/PaddleOCR Fallback
"""

import os
import sys
import re
import unicodedata
from typing import List, Dict, Tuple, Sequence

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC và làm sạch khoảng trắng."""
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def levenshtein_distance(s1: Sequence, s2: Sequence) -> int:
    """Tính khoảng cách Levenshtein giữa 2 chuỗi hoặc 2 danh sách từ (tokens)."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def calculate_substring_cer(detected_full: str, target: str) -> Tuple[float, str]:
    """
    Tính Character Error Rate (CER) bằng cách tìm Substring Window tốt nhất trong detected_full
    khớp với target phrase, thay vì so sánh toàn bộ bức poster với 1 cụm từ ngắn.
    """
    pred_norm = normalize_text(detected_full).upper()
    gt_norm = normalize_text(target).upper()
    
    if not gt_norm:
        return (0.0 if not pred_norm else 1.0), ""
        
    if not pred_norm:
        return 1.0, ""

    # Nếu target nằm trọn trong pred_norm -> CER = 0
    if gt_norm in pred_norm:
        return 0.0, gt_norm

    # Dùng Sliding Window với kích thước tương đương len(gt_norm) ± 30%
    gt_len = len(gt_norm)
    min_dist = float('inf')
    best_sub = ""

    window_sizes = set([gt_len, max(1, int(gt_len * 0.8)), int(gt_len * 1.2), int(gt_len * 1.5)])
    
    for w in window_sizes:
        if w > len(pred_norm) + 5:
            continue
        for i in range(0, max(1, len(pred_norm) - w + 1)):
            sub = pred_norm[i:i + w]
            dist = levenshtein_distance(sub, gt_norm)
            if dist < min_dist:
                min_dist = dist
                best_sub = sub

    # Nếu không tìm thấy cửa sổ hợp lý, lấy khoảng cách trực tiếp nhưng bounded
    if min_dist == float('inf'):
        min_dist = levenshtein_distance(pred_norm, gt_norm)

    cer = round(min_dist / max(len(gt_norm), 1), 4)
    return min(cer, 1.0), best_sub

def calculate_cer(pred: str, gt: str) -> float:
    """Wrapper tương thích ngược tính CER."""
    cer, _ = calculate_substring_cer(pred, gt)
    return cer

def calculate_wer(pred_tokens: List[str], gt_tokens: List[str]) -> float:
    """Tính Word Error Rate (WER) chuẩn trên danh sách từ."""
    if not gt_tokens:
        return 0.0 if not pred_tokens else 1.0
    if not pred_tokens:
        return 1.0
    dist = levenshtein_distance(pred_tokens, gt_tokens)
    return round(min(dist / len(gt_tokens), 1.0), 4)

def calculate_ned(pred: str, gt: str) -> float:
    """Normalized Edit Distance (NED = 1 - (dist / max_len))."""
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    max_len = max(len(pred_norm), len(gt_norm))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(pred_norm, gt_norm)
    return round(1.0 - (dist / max_len), 4)

# Singletons cho OCR models
_vietocr_predictor = None

def get_vietocr_predictor():
    global _vietocr_predictor
    if _vietocr_predictor is None:
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
            import torch
            
            config = Cfg.load_config_from_name('vgg_transformer')
            config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            config['predictor']['beamsearch'] = False
            
            _vietocr_predictor = Predictor(config)
            print("🟢 Nạp thành công mô hình VietOCR Predictor!")
        except Exception as e:
            print(f"⚠️ Chưa thể khởi tạo VietOCR: {e}")
            _vietocr_predictor = False
    return _vietocr_predictor

def detect_and_crop_text_regions(image_path: str) -> List:
    """
    Sử dụng OpenCV để phát hiện các vùng văn bản (Bounding Boxes) trên poster
    và crop ra từng đoạn chữ trước khi đưa vào VietOCR.
    """
    crops = []
    try:
        import cv2
        from PIL import Image

        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return crops

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # Gradient bộc lộ ranh giới nét chữ
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Kết nối các ký tự gần nhau thành dòng chữ
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pil_img = Image.open(image_path).convert('RGB')
        w_img, h_img = pil_img.size

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Lọc bớt box nhiễu quá nhỏ hoặc quá to
            if w > 20 and h > 10 and (w * h) < (w_img * h_img * 0.8):
                # Thêm padding xung quanh box
                pad = 4
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w_img, x + w + pad)
                y2 = min(h_img, y + h + pad)
                crop = pil_img.crop((x1, y1, x2, y2))
                crops.append(crop)
    except Exception as e:
        print(f"Lỗi khi crop text region bằng OpenCV: {e}")

    # Nếu không detect được contour nào, fallback trả về nguyên bức ảnh
    if not crops and os.path.exists(image_path):
        from PIL import Image
        crops.append(Image.open(image_path).convert('RGB'))
        
    return crops

def recognize_image_text(image_path: str) -> str:
    """Đọc chữ Tiếng Việt từ file ảnh với VietOCR (Cropped regions)."""
    if not image_path or not os.path.exists(image_path):
        return ""
        
    predictor = get_vietocr_predictor()
    detected_lines = []

    if predictor:
        try:
            crops = detect_and_crop_text_regions(image_path)
            for crop_img in crops:
                line_text = predictor.predict(crop_img)
                if line_text and line_text.strip():
                    detected_lines.append(line_text.strip())
        except Exception as e:
            print(f"Lỗi khi VietOCR đọc ảnh {image_path}: {e}")

    return " ".join(detected_lines)

def evaluate_texts(detected_text: str, required_texts: list, image_path: str = None) -> dict:
    """
    Đánh giá độ chính xác chữ (CER, WER, Exact Match Ratio, NED) của các required_texts.
    """
    if image_path and os.path.exists(image_path):
        ocr_text = recognize_image_text(image_path)
        if ocr_text:
            detected_text = ocr_text
            
    pred_norm = normalize_text(detected_text).upper()
    
    total_cer = 0.0
    total_wer = 0.0
    total_ned = 0.0
    exact_matches = 0
    
    num_reqs = max(len(required_texts), 1)

    for req in required_texts:
        req_norm = normalize_text(req).upper()
        if req_norm in pred_norm:
            exact_matches += 1
            total_ned += 1.0
            total_cer += 0.0
            total_wer += 0.0
        else:
            cer, best_sub = calculate_substring_cer(pred_norm, req_norm)
            ned = calculate_ned(best_sub if best_sub else pred_norm, req_norm)
            
            # Wer trên tokens
            pred_tokens = best_sub.split() if best_sub else pred_norm.split()
            gt_tokens = req_norm.split()
            wer = calculate_wer(pred_tokens, gt_tokens)

            total_cer += cer
            total_wer += wer
            total_ned += ned
            
    avg_cer = round(total_cer / num_reqs, 4)
    avg_wer = round(total_wer / num_reqs, 4)
    avg_ned = round(total_ned / num_reqs, 4)
    exact_ratio = round(exact_matches / num_reqs, 4)
    
    # Điều kiện fail cứng: CER > 0.30 hoặc không exact match dòng nào khi required_texts có chữ
    is_fail = (avg_cer > 0.30) or (exact_matches == 0 and len(required_texts) > 0)
    
    return {
        "detected_text": detected_text,
        "cer": avg_cer,
        "wer": avg_wer,
        "ned": avg_ned,
        "exact_match_ratio": exact_ratio,
        "is_fail_text": is_fail
    }

if __name__ == "__main__":
    gt = "THỜI GIAN LÀ CỦA BẠN"
    pred = "DÒNG ĐỒNG HỒ CAO CẤP THỜI GIAN LÀ CỦA BAN NÂNG TẦM PHONG CÁCH"
    res = evaluate_texts(detected_text=pred, required_texts=[gt])
    print(f"Pred Full Text: {pred}")
    print(f"Target Text   : {gt}")
    print(f"CER: {res['cer']} | WER: {res['wer']} | NED: {res['ned']} | Exact: {res['exact_match_ratio']}")
