"""
Build the deterministic TendooBizEval-Vi v1 case files using real Unsplash Lifestyle Product metadata.
Strictly enforces FLUX.2 Klein 4B Specifications:
- Dimensions are all multiples of 16 (1024x1024, 1024x1280, 1280x720, 1024x1536, 1024x768, 1088x1920, 1200x624)
- Prompts are natural, human-like prompts WITHOUT internal system file paths.
- Diverse text length (short, medium, long)
"""

from __future__ import annotations

import json
import sys
from math import gcd
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "benchmarks" / "tendoo_v0" / "cases"
MANIFEST_PATH = ROOT / "benchmarks" / "tendoo_v0" / "manifests" / "reference_manifest.json"

SIZES = [
    (1024, 1024),
    (1024, 1280),
    (1280, 720),
    (1024, 1536),
    (1024, 768),
    (1088, 1920),
    (1200, 624)
]

LAYOUTS = [
    "title_top_cta_bottom",
    "title_left_product_right",
    "product_center_text_around",
    "before_after_split",
    "text_block_below_product",
    "full_bleed_cta_corner"
]

EDIT_TYPES = [
    "background_replacement",
    "lifestyle_placement",
    "key_visual",
    "preserve_packaging_logo",
    "object_removal_cleanup"
]

# 50 Real-world Tendoo Business Prompts
T2I_DOMAIN_PROMPTS = [
    ("food_beverage", "Phở Bò Gia Truyền", "Phở bò Hà Nội nghi ngút khói trên bàn gỗ cổ điển, kèm quẩy giòn và ớt tươi", "PHỞ BÒ GIA TRUYỀN", "ĐẬM VỊ NƯỚC DÙNG 24H", "GIÁ CHỈ 55.000đ", "product_hero"),
    ("food_beverage", "Cà Phê Phin Việt Nam", "Ly cà phê sữa đá phin truyền thống chảy từng giọt vàng óng trên quầy gỗ cafe xưa", "CÀ PHÊ PHIN NGUYÊN CHẤT", "ĐẬM VỊ ĐẤT VIỆT TẠI QUÁN CÀ PHÊ PHIN TRUYỀN THỐNG CHUẨN VỊ HÀ NỘI", "MUA 2 TẶNG 1 HÔM NAY CHO KHÁCH HÀNG MỚI", "product_hero"),
    ("food_beverage", "Trà Sữa Giải Nhiệt", "Ly trà sữa trân trùng đường đen mát lạnh có lớp kem cheese béo ngậy trên nền nhiệt đới", "TRÀ SỮA ĐỒNG GIÁ 25K", "TROPICAL TEA BOUTIQUE", "GIẢM 30% KHI ĐẶT APP", "flash_sale"),
    ("food_beverage", "Bánh Mì Sài Gòn", "Ổ bánh mì thịt nướng giòn rụm với rau dưa tươi ngon trên giấy báo vintage", "BÁNH MÌ SÀI GÒN GIÒN RỤM", "THE VIET BÁNH MÌ", "COMBO SÁNG CHỈ 35.000đ", "lifestyle"),
    ("food_beverage", "Bún Chả Hà Nội", "Mẹt bún chả nướng than hoa thơm lừng với bát nước chấm tỏi ớt truyền thống", "BÚN CHẢ HÀ NỘI CHUẨN VỊ", "BÚN CHẢ MỆT XƯA", "GIAO TẬN NƠI TỪ 45.000đ TẠI QUẬN HOÀN KIẾM HÀ NỘI HÔM NAY", "informational"),
    ("food_beverage", "Menu Thực Đơn Quán Nước", "Menu thực đơn quán nước hiện đại có hình ảnh trà trái cây tươi mát và trân châu", "MENU TRÀ", "TENDO", "29K", "informational"),

    ("events_recruitment", "Banner Mừng Khai Trương", "Banner quảng cáo khai trương rực rỡ sắc đỏ vàng với bóng bay và lẵng hoa tươi sang trọng", "TƯNG BỪNG KHAI TRƯƠNG", "SHOP THỜI TRANG TENDOOBIZ", "GIẢM 50% TOÀN BỘ CỬA HÀNG", "seasonal"),
    ("events_recruitment", "Poster Tuyển Dụng Nhân Viên", "Poster tuyển dụng nhân viên bán hàng hiện đại năng động với icon văn phòng sáng tạo", "TENDOOSHOP TUYỂN DỤNG", "NHÂN VIÊN BÁN HÀNG TOÀN THỜI GIAN VỚI MỨC THU NHẬP HẤP DẪN VÀ THƯỞNG DOANH SỐ", "THU NHẬP 8-12 TRIỆU/THÁNG", "informational"),
    ("events_recruitment", "Voucher Khuyến Mãi Mới", "Voucher ưu đãi thiết kế sang trọng tone vàng kim champagne với tem giảm giá", "VOUCHER VIP", "TENDOO", "GIẢM 200K", "flash_sale"),
    ("events_recruitment", "Mừng Sinh Nhật Cửa Hàng", "Banner mừng sinh nhật cửa hàng 5 tuổi rực rỡ đèn pháo hoa và hộp quà may mắn", "MỪNG SINH NHẬT 5 TUỔI", "TENDOO RETAIL STORE", "BỐC THĂM TRÚNG QUÀ 10 TRIỆU", "seasonal"),

    ("feedback_social_proof", "Ảnh Feedback Khách Hàng Spa", "Khung ảnh feedback chăm sóc da spa với biểu tượng 5 sao vàng và gương mặt rạng rỡ", "KHÁCH HÀNG NÓI GÌ SAU SPA?", "DERMA BEAUTY CARE", "ĐÁNH GIÁ 5/5 SAO TỪ KHÁCH HÀNG TOÀN QUỐC HÀI LÒNG 100%", "testimonial"),
    ("feedback_social_proof", "Review Chăm Sóc Thú Cung", "Khung ảnh review em cún Poodle xinh xắn sau khi spa cắt tỉa với đánh giá hài lòng", "BOSS LỘT XÁC SAU 2 GIỜ", "PET SPA & GROOMING", "KHÁCH HÀNG HÀI LÒNG 100%", "testimonial"),
    ("feedback_social_proof", "Hình Ảnh Feedback Tập Gym", "Ảnh biến đổi vóc dáng săn chắc sau 90 ngày tập luyện với private coach", "90 NGÀY THAY ĐỔI VÓC DÁNG", "PRIVATE COACHING 1:1", "TẶNG BUỔI ĐÁNH GIÁ THỂ LỰC", "before_after"),
    ("feedback_social_proof", "Cảm Ơn Khách Hàng Thân Thiết", "Banner lời cảm ơn chân thành từ thương hiệu gửi tới 10.000 khách hàng đã tin dùng", "CẢM ƠN VIP", "TENDOO", "GIẢM 20%", "testimonial"),

    ("vietnamese_culture", "Tết Cổ Truyền Việt Nam", "Không gian ngày Tết Việt Nam với hoa mai vàng, hoa đào hồng, bánh chưng và câu đối đỏ", "MỪNG XUÂN BÍNH NGỌ 2026", "TẾT ĐOÀN VIÊN AN KHANG HẠNH PHÚC BÊN GIA ĐÌNH VÀ NGUYỆN CẦU BÌNH AN", "QUÀ TẾT CAO CẤP GIẢM 30%", "seasonal"),
    ("vietnamese_culture", "Áo Dài Truyền Thống", "Thiếu nữ Việt Nam duyên dáng trong chiếc áo dài lụa hồng bên nón lá và hoa sen trắng", "ÁO DÀI VIỆT NAM DUYÊN DÁNG", "BST LỤA HÀ ĐÔNG 2026", "ƯU ĐÃI 15% ĐẶT MAY SỚM", "lifestyle"),
    ("vietnamese_culture", "Nón Lá & Quê Hương", "Bối cảnh làng quê Việt Nam thanh bình với chiếc nón lá truyền thống và cánh đồng lúa chín vàng", "HƯƠNG VỊ VIỆT", "ĐẶC SẢN VIỆT", "FREESHIP", "product_hero"),

    ("consumer_electronics", "Tai Nghe Không Dây", "Tai nghe bluetooth cao cấp vỏ kim loại mờ nhám trên kệ đèn neon hiện đại", "ÂM THANH TRONG TẦM TAY", "AUDIOFLOW PRO 2026", "GIẢM 25% HÔM NAY", "product_hero"),
    ("beauty_health", "Serum Dưỡng Da", "Chai serum thủy tinh trong suốt với những giọt tinh chất căng mọng bên lá aloe vera", "LÀN DA KHỎE TỪ BÊN TRONG", "PURE DERMA SERUM 30ML TỰ NHIÊN DƯỠNG ẨM CHUYÊN SÂU", "GIẢM 30% ĐƠN ĐẦU", "product_hero"),
    ("home_decor", "Sofa Phòng Khách Smart", "Bộ ghế sofa phòng khách phong cách Zen tối giản ấm áp ánh nắng chiều", "SỰ THƯ GIÃN", "SMART SOFA", "TẶNG BÀN TRÀ", "product_hero")
]

def aspect(size):
    d = gcd(*size)
    return f"{size[0] // d}:{size[1] // d}"

def text_meta(texts, language_mix):
    longest = max(map(len, texts))
    length = "short" if longest <= 16 else "medium" if longest <= 40 else "long"
    types = ["headline"] + (["product_name"] if len(texts) > 1 else []) + (["promotion"] if len(texts) > 2 else [])
    numeric = "currency" if any("đ" in text for text in texts) else "percentage" if any("%" in text for text in texts) else "date_or_quantity" if any(char.isdigit() for text in texts for char in text) else "none"
    return {"text_length": length, "text_block_count": len(texts), "text_type": types, "language_mix": language_mix, "numeric_pattern": numeric}

def load_manifest_references():
    if MANIFEST_PATH.exists():
        try:
            data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            return data.get("references", [])
        except Exception:
            pass
    return []

def make_t2i():
    rows = []
    total_configs = (T2I_DOMAIN_PROMPTS * 3)[:50]
    
    for i, config in enumerate(total_configs, 1):
        category, label, product, headline, name, offer, style = config
        layout, size = LAYOUTS[(i - 1) % len(LAYOUTS)], SIZES[(i - 1) % len(SIZES)]
        
        texts = [headline, name, offer]
        instruction = (
            f"Thiết kế {style} quảng cáo thương mại cho {product} thuộc nhóm {label}. "
            f"Bối cảnh và ánh sáng phù hợp ngành hàng, sản phẩm là chủ thể rõ nét, bố cục {layout}. "
            f"Hiển thị chính xác ba dòng chữ: {', '.join(repr(text) for text in texts)}. "
            f"Hình ảnh chân thực, typography dễ đọc, không thêm chữ giả, kích thước chuẩn FLUX.2 {size[0]}x{size[1]} ({aspect(size)})."
        )
        
        rows.append({
            "case_id": f"t2i_{i:03d}",
            "track": "t2i",
            "reference_image": None,
            "instruction": instruction,
            "required_text": texts,
            "product_attributes": {
                "category": category,
                "color": "specified_by_prompt",
                "preserve_logo": False,
                "preserve_shape": False
            },
            "edit_type": None,
            "target_layout": layout,
            "difficulty": ["easy", "medium", "hard"][(i - 1) % 3],
            "output_size": list(size),
            **text_meta(texts, "vietnamese_only" if i % 4 != 0 else "vietnamese_plus_english")
        })
    return rows

def make_i2i():
    rows = []
    manifest_refs = load_manifest_references()
    
    for i in range(1, 51):
        ref_item = manifest_refs[i - 1] if i - 1 < len(manifest_refs) else {}
        ref_path = ref_item.get("path") or f"benchmarks/tendoo_v0/references/simple_product/prod_{i:03d}.png"
        prod_title = ref_item.get("title") or f"Sản phẩm Lifestyle #{i:03d}"
        prod_cat = ref_item.get("category") or "general"
        
        edit_type = EDIT_TYPES[(i - 1) % len(EDIT_TYPES)]
        size = SIZES[(i - 1) % len(SIZES)]
        layout = LAYOUTS[(i - 1) % len(LAYOUTS)]
        
        if i % 6 == 1:
            # Short text
            texts = ["HOT SALE", "GIẢM 20%"]
            bối_cảnh = "bối cảnh studio hiện đại tone màu pastel sang trọng"
        elif i % 6 == 2:
            # Medium text
            texts = [f"MUA NGAY - {prod_title.upper()[:20]}", "MUA 1 TẶNG 1 HÔM NAY"]
            bối_cảnh = "quầy trưng bày bán lẻ cao cấp ánh sáng nịnh mắt"
        elif i % 6 == 3:
            # Short text
            texts = ["KHAI TRƯƠNG", "GIẢM 50%"]
            bối_cảnh = "không gian cửa hàng khai trương rực rỡ bóng bay"
        elif i % 6 == 4:
            # Medium text
            texts = ["LÀN DA KHỎE", f"SẢN PHẨM {prod_title.upper()[:15]}", "ĐÁNH GIÁ 5/5 SAO"]
            bối_cảnh = "bối cảnh chụp hình sản phẩm commercial cao cấp"
        elif i % 6 == 5:
            # Long text
            texts = ["MỪNG XUÂN BÍNH NGỌ 2026 TẾT ĐOÀN VIÊN HẠNH PHÚC BÊN GIA ĐÌNH VÀ NGUYỆN CẦU AN KHANG", f"QUÀ TẾT {prod_title.upper()[:15]} GIẢM 30%"]
            bối_cảnh = "phông nền Tết cổ truyền Việt Nam rực rỡ sắc xuân"
        else:
            # Long text
            texts = ["THÔNG BÁO CHƯƠNG TRÌNH TRI ẦN KHÁCH HÀNG VIP LỚN NHẤT TRONG NĂM 2026", "TẶNG NGAY VOUCHER 500K TOÀN BỘ CỬA HÀNG"]
            bối_cảnh = "bối cảnh thương mại cao cấp với banner nghệ thuật"
            
        preserve = "giữ nguyên logo, màu sắc, hình dáng và các chi tiết nhận diện sản phẩm"
        
        instruction = (
            f"Chỉnh sửa ảnh reference của sản phẩm {prod_title} (ngành hàng {prod_cat}). "
            f"Thực hiện tác vụ {edit_type}: đặt sản phẩm vào {bối_cảnh}. {preserve}. "
            f"Bố cục {layout}; hiển thị chính xác các dòng chữ: {', '.join(repr(text) for text in texts)}. "
            f"Không tạo sản phẩm khác, không thêm chữ giả, kích thước chuẩn FLUX.2 {size[0]}x{size[1]} ({aspect(size)})."
        )
        
        rows.append({
            "case_id": f"i2i_{i:03d}",
            "track": "i2i",
            "reference_image": ref_path,
            "instruction": instruction,
            "required_text": texts,
            "product_attributes": {
                "category": prod_cat,
                "product_name": prod_title,
                "color": "original",
                "preserve_logo": True,
                "preserve_shape": True
            },
            "edit_type": edit_type,
            "target_layout": layout,
            "difficulty": ["easy", "medium", "hard"][(i - 1) % 3],
            "output_size": list(size),
            **text_meta(texts, "vietnamese_only" if i % 3 != 0 else "vietnamese_plus_english")
        })
    return rows

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

def build_benchmark_v1():
    t2i, i2i = make_t2i(), make_i2i()
    write_jsonl(CASE_DIR / "t2i.jsonl", t2i)
    write_jsonl(CASE_DIR / "i2i.jsonl", i2i)
    print(f"Đã ghi {len(t2i)} T2I và {len(i2i)} I2I cases chuẩn FLUX.2 Specs (Prompts tự nhiên 100%) vào {CASE_DIR}")

if __name__ == "__main__":
    build_benchmark_v1()
