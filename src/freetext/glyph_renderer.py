"""
Vietnamese Glyph Rendering Engine for FreeText.
Supports UTF-8 Vietnamese characters, multi-line typography, and layout positioning.
"""

import re
import os
from typing import List, Tuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch


def extract_text_spans(prompt: str) -> List[str]:
    """
    Extract target text spans from a prompt.
    Looks for text enclosed in quotes: "...", '...', “...”, or 「...」
    """
    patterns = [
        r'"([^"]+)"',
        r"'([^']+)'",
        r'“([^”]+)”',
        r'‘([^’]+)’',
        r'「([^」]+)」',
    ]
    spans = []
    for pat in patterns:
        matches = re.findall(pat, prompt)
        for m in matches:
            cleaned = m.strip()
            if cleaned and cleaned not in spans:
                spans.append(cleaned)
    return spans


def get_vietnamese_font(font_size: int = 48, preferred_font_path: Optional[str] = None) -> ImageFont.ImageFont:
    """
    Load a font that supports Vietnamese diacritics.
    Checks preferred path, Windows system fonts, Linux fonts, and fallback.
    """
    candidates = []
    if preferred_font_path and os.path.exists(preferred_font_path):
        candidates.append(preferred_font_path)

    # Windows fonts
    candidates.extend([
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ])

    # Linux fonts (for server execution)
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    ])

    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=font_size)
            except Exception:
                continue

    # Default fallback
    try:
        return ImageFont.load_default()
    except Exception:
        return None


class GlyphRenderer:
    """
    Renders text strings into a high-contrast glyph reference image (I_glyph)
    with accurate bounding boxes and spatial masks.
    """
    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path

    def render_text_canvas(
        self,
        texts: List[str],
        width: int = 1024,
        height: int = 1024,
        box_coords: Optional[List[Tuple[int, int, int, int]]] = None,
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Tuple[Image.Image, np.ndarray, List[Dict]]:
        """
        Renders a list of text strings onto an image canvas.

        :param texts: List of text strings to render
        :param width: Image width
        :param height: Image height
        :param box_coords: Optional predefined bounding boxes [(x1, y1, x2, y2), ...]
        :return: (PIL Image, Binary Mask ndarray [H, W], List of region dicts)
        """
        img = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        mask = np.zeros((height, width), dtype=np.float32)
        regions = []

        if not texts:
            return img, mask, regions

        num_texts = len(texts)

        for i, text in enumerate(texts):
            if not text.strip():
                continue

            if box_coords and i < len(box_coords):
                x1, y1, x2, y2 = box_coords[i]
            else:
                # Automatic balanced banner layout
                # Header / middle / footer placement
                slot_h = height // (num_texts + 1)
                center_y = slot_h * (i + 1)
                margin_x = int(width * 0.1)
                box_w = width - 2 * margin_x
                box_h = int(slot_h * 0.8)
                x1 = margin_x
                y1 = center_y - box_h // 2
                x2 = width - margin_x
                y2 = center_y + box_h // 2

            bw = max(x2 - x1, 10)
            bh = max(y2 - y1, 10)

            # Determine optimal font size
            font_size = max(12, int(bh * 0.65))
            font = get_vietnamese_font(font_size=font_size, preferred_font_path=self.font_path)

            # Fit text within bounding box
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                while tw > bw and font_size > 14:
                    font_size = int(font_size * 0.85)
                    font = get_vietnamese_font(font_size=font_size, preferred_font_path=self.font_path)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
            except Exception:
                tw, th = bw // 2, bh // 2

            tx = x1 + (bw - tw) // 2
            ty = y1 + (bh - th) // 2

            draw.text((tx, ty), text, fill=text_color, font=font)

            # Bounding box for mask with slight padding
            pad_x = int(tw * 0.05) + 4
            pad_y = int(th * 0.08) + 4
            rx1 = max(0, tx - pad_x)
            ry1 = max(0, ty - pad_y)
            rx2 = min(width, tx + tw + pad_x)
            ry2 = min(height, ty + th + pad_y)

            mask[ry1:ry2, rx1:rx2] = 1.0

            regions.append({
                "text": text,
                "box": (rx1, ry1, rx2, ry2),
                "text_pos": (tx, ty),
                "font_size": font_size,
            })

        return img, mask, regions

    def get_glyph_tensor(
        self,
        texts: List[str],
        width: int = 1024,
        height: int = 1024,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Returns normalized glyph image tensor [1, 3, H, W] in [-1, 1] range
        and binary mask tensor [1, 1, H, W] in [0, 1].
        """
        img, mask_np, regions = self.render_text_canvas(texts, width=width, height=height)

        img_np = np.array(img, dtype=np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)

        return img_tensor, mask_tensor, regions
