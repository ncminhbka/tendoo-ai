"""Vietnamese OCR and text metrics for the Tendoo benchmark."""

import os
import re
import sys
import unicodedata
from typing import List, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_paddle_ocr = None
_paddle_failed = False


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def levenshtein_distance(s1: Sequence, s2: Sequence) -> int:
    previous = list(range(len(s2) + 1))
    for i, left in enumerate(s1, 1):
        current = [i]
        for j, right in enumerate(s2, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def calculate_substring_cer(detected_full: str, target: str) -> Tuple[float, str]:
    pred = normalize_text(detected_full).upper()
    gold = normalize_text(target).upper()
    if not gold:
        return (0.0 if not pred else 1.0), ""
    if not pred:
        return 1.0, ""
    if gold in pred:
        return 0.0, gold
    sizes = {len(gold), max(1, int(len(gold) * 0.8)),
             max(1, int(len(gold) * 1.2)), max(1, int(len(gold) * 1.5))}
    best_distance, best_sub = float("inf"), ""
    for size in sizes:
        for index in range(max(1, len(pred) - size + 1)):
            candidate = pred[index:index + size]
            distance = levenshtein_distance(candidate, gold)
            if distance < best_distance:
                best_distance, best_sub = distance, candidate
    if best_distance == float("inf"):
        best_distance, best_sub = levenshtein_distance(pred, gold), pred
    return min(round(best_distance / max(len(gold), 1), 4), 1.0), best_sub


def calculate_cer(pred: str, gt: str) -> float:
    return calculate_substring_cer(pred, gt)[0]


def calculate_wer(pred_tokens: List[str], gt_tokens: List[str]) -> float:
    if not gt_tokens:
        return 0.0 if not pred_tokens else 1.0
    if not pred_tokens:
        return 1.0
    return round(min(levenshtein_distance(pred_tokens, gt_tokens) / len(gt_tokens), 1.0), 4)


def calculate_ned(pred: str, gt: str) -> float:
    pred_norm, gt_norm = normalize_text(pred), normalize_text(gt)
    max_len = max(len(pred_norm), len(gt_norm))
    if max_len == 0:
        return 1.0
    return round(1.0 - levenshtein_distance(pred_norm, gt_norm) / max_len, 4)


def get_paddle_ocr():
    global _paddle_ocr, _paddle_failed
    if _paddle_ocr is None and not _paddle_failed:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(
                lang="vi",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            print("[benchmark] Loaded PaddleOCR Vietnamese backend.")
        except Exception as exc:
            _paddle_failed = True
            print(f"[benchmark] PaddleOCR unavailable: {exc}")
    return _paddle_ocr


def recognize_image_text(image_path: str) -> tuple[str, str]:
    """Read text and return ``(text, engine_status)``.

    ``measured`` means PaddleOCR ran successfully, including when it detected
    zero text. That distinction is essential: an empty measured result is a
    real model failure and must contribute CER=1/NED=0 to the benchmark.
    """
    if not image_path or not os.path.exists(image_path):
        return "", "no_image"
    ocr = get_paddle_ocr()
    if ocr is None:
        return "", "unavailable"
    try:
        result = ocr.predict(image_path) if hasattr(ocr, "predict") else ocr.ocr(image_path, cls=True)
        detected = []
        for page in result or []:
            if isinstance(page, dict):
                detected.extend(str(value).strip() for value in page.get("rec_texts", []) if str(value).strip())
            else:
                for line in page or []:
                    if len(line) > 1 and isinstance(line[1], (list, tuple)):
                        value = str(line[1][0]).strip()
                        if value:
                            detected.append(value)
        return " ".join(detected), "measured"
    except Exception as exc:
        print(f"[benchmark] PaddleOCR failed for {image_path}: {exc}")
        return "", "error"


def evaluate_texts(detected_text: str, required_texts: list, image_path: str = None) -> dict:
    ocr_used = bool(image_path and os.path.exists(image_path))
    ocr_status = "manual" if not image_path else "no_image"
    if ocr_used:
        ocr_text, ocr_status = recognize_image_text(image_path)
        if ocr_status == "measured":
            detected_text = ocr_text

    prediction = normalize_text(detected_text).upper()
    total_cer = total_wer = total_ned = 0.0
    exact_matches = 0
    count = max(len(required_texts), 1)
    for required in required_texts:
        target = normalize_text(required).upper()
        if target in prediction:
            exact_matches += 1
            total_ned += 1.0
            continue
        cer, best_sub = calculate_substring_cer(prediction, target)
        total_cer += cer
        total_ned += calculate_ned(best_sub or prediction, target)
        total_wer += calculate_wer((best_sub or prediction).split(), target.split())

    avg_cer = round(total_cer / count, 4)
    exact_ratio = round(exact_matches / count, 4)
    return {
        "detected_text": detected_text,
        "ocr_status": ocr_status,
        "ocr_method": "PaddleOCR-vi",
        "cer": avg_cer,
        "wer": round(total_wer / count, 4),
        "ned": round(total_ned / count, 4),
        "exact_match_ratio": exact_ratio,
        "is_fail_text": bool(required_texts) and (avg_cer > 0.30 or exact_matches == 0),
    }
