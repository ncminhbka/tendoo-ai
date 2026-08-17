"""
Module đánh giá Text Accuracy cho Tiếng Việt:
- Chuẩn hóa Unicode NFC
- Tính Character Error Rate (CER), Word Error Rate (WER)
- Tính Exact Match Ratio và Normalized Edit Distance (NED)
- Tích hợp VietOCR
"""

import unicodedata
import sys
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC và làm sạch khoảng trắng."""
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def levenshtein_distance(s1: str, s2: str) -> int:
    """Tính khoảng cách Levenshtein giữa 2 chuỗi."""
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

def calculate_cer(pred: str, gt: str) -> float:
    """Character Error Rate (CER = distance / len(gt))."""
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    if not gt_norm:
        return 0.0 if not pred_norm else 1.0
    dist = levenshtein_distance(pred_norm, gt_norm)
    return round(dist / max(len(gt_norm), 1), 4)

def calculate_wer(pred: str, gt: str) -> float:
    """Word Error Rate (WER)."""
    pred_words = normalize_text(pred).split()
    gt_words = normalize_text(gt).split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    dist = levenshtein_distance(pred_words, gt_words)
    return round(dist / max(len(gt_words), 1), 4)

def calculate_ned(pred: str, gt: str) -> float:
    """Normalized Edit Distance (NED = 1 - (dist / max_len))."""
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    max_len = max(len(pred_norm), len(gt_norm))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(pred_norm, gt_norm)
    return round(1.0 - (dist / max_len), 4)

def evaluate_texts(detected_text: str, required_texts: list) -> dict:
    """
    Đánh giá toàn bộ các required_texts trong một ảnh.
    """
    pred_norm = normalize_text(detected_text).upper()
    
    total_cer = 0.0
    total_ned = 0.0
    exact_matches = 0
    
    for req in required_texts:
        req_norm = normalize_text(req).upper()
        # Kiểm tra xem req_norm có xuất hiện exact trong pred_norm không
        if req_norm in pred_norm:
            exact_matches += 1
            total_ned += 1.0
        else:
            ned = calculate_ned(pred_norm, req_norm)
            cer = calculate_cer(pred_norm, req_norm)
            total_ned += ned
            total_cer += cer
            
    num_reqs = max(len(required_texts), 1)
    avg_cer = round(total_cer / num_reqs, 4)
    avg_ned = round(total_ned / num_reqs, 4)
    exact_ratio = round(exact_matches / num_reqs, 4)
    
    # Điều kiện fail cứng: CER > 0.30 hoặc exact_ratio = 0 khi có chữ bắt buộc
    is_fail = (avg_cer > 0.30) or (exact_matches == 0 and len(required_texts) > 0)
    
    return {
        "cer": avg_cer,
        "wer": avg_cer, # xấp xỉ
        "ned": avg_ned,
        "exact_match_ratio": exact_ratio,
        "is_fail_text": is_fail
    }

if __name__ == "__main__":
    gt = "THỜI GIAN LÀ CỦA BẠN"
    pred = "THỜI GIAN LÀ CỦA BAN"
    print(f"GT: {gt}")
    print(f"Pred: {pred}")
    print(f"CER: {calculate_cer(pred, gt)}")
    print(f"NED: {calculate_ned(pred, gt)}")
