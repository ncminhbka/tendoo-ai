"""
Script chuyển đổi toàn bộ prompt_test.txt và dữ liệu Tendoo thành 2 file dataset chuẩn JSONL:
- benchmarks/tendoo_v0/cases/t2i.jsonl (50 cases T2I)
- benchmarks/tendoo_v0/cases/i2i.jsonl (50 cases I2I)
"""

import json
import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def generate_cases():
    os.makedirs("benchmarks/tendoo_v0/cases", exist_ok=True)
    
    # ── 1. ĐỌC VÀ PHÂN TÍCH PROMPT_TEST.TXT ĐỂ TẠO T2I.JSONL (50 CASES) ──
    t2i_cases = []
    
    prompt_file = "prompt_test.txt"
    raw_prompts = []
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Tách các prompt bằng dòng trống hoặc xuống dòng
            raw_prompts = [p.strip() for p in content.split("\n\n") if p.strip()]

    # Mẫu danh sách required_texts cho các prompt
    t2i_definitions = [
        ("t2i_001", "Dong_Ho", ["THỜI GIAN LÀ CỦA BẠN", "NÂNG TẦM PHONG CÁCH ĐỜI SỐNG"], [1024, 1280]),
        ("t2i_002", "Dong_Ho", ["CHINH PHỤC MỌI GIỚI HẠN", "DÒNG ĐỒNG HỒ THỂ THAO CAO CẤP"], [1024, 1820]),
        ("t2i_003", "Dong_Ho", ["KẾT NỐI YÊU THƯƠNG", "NGƯỜI BẠN ĐỒNG HÀNH CỦA GIA ĐÌNH"], [1024, 1024]),
        ("t2i_004", "Dong_Ho", ["ĐẲNG CẤP & SANG TRỌNG", "BIỂU TƯỢNG THỜI TRANG MỚI"], [1024, 1536]),
        ("t2i_005", "Dong_Ho", ["TƯƠNG LAI TRONG TẦM TAY", "KHÁM PHÁ CÔNG NGHỆ ĐỘT PHÁ"], [1024, 1820]),
        ("t2i_006", "Dong_Ho", ["SỰ LỰA CHỌN CỦA NHÀ LÃNH ĐẠO", "NÂNG TẦM THÀNH CÔNG"], [1024, 768]),
        ("t2i_007", "Dong_Ho", ["SỰ TINH TẾ CỦA SỨC MẠNH"], [1280, 720]),
        ("t2i_008", "Dong_Ho", ["CHỐNG NƯỚC 50M", "SẴN SÀNG CHO MỌI CUỘC PHIÊU LƯU"], [1024, 1024]),
        ("t2i_009", "Dong_Ho", ["NGĂN NẮP & HIỆN ĐẠI", "NHỮNG VẬT BẤT LY THÂN CỦA BẠN"], [1024, 1536]),
        ("t2i_010", "Dong_Ho", ["KHÁM PHÁ THẾ GIỚI CÙNG BẠN", "NGƯỜI BẠN ĐỒNG HÀNH TRÊN MỌI NẺO ĐƯỜNG"], [1280, 720]),
        ("t2i_011", "Fitness_Gym", ["KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY THAY ĐỔI?", "PRIVATE COACHING TRANSFORMATION", "GIẢM 20% GÓI PT THÁNG ĐẦU"], [1024, 1024]),
        ("t2i_012", "Spa_Thu_Cung", ["BOSS LỘT XÁC THẾ NÀO SAU 2 GIỜ AT SPA?", "PREMIUM PET GROOMING & SPA", "TẶNG GÓI NGÂM SỤC OZON 200K"], [1024, 1024]),
        ("t2i_013", "Du_Lich_Glamping", ["TRẢI NGHIỆM CHỮA LÀNH GIỮA THIÊN NHIÊN", "CLOUD RETREAT GLAMPING", "GIẢM 30% KHI ĐẶT PHÒNG SỚM"], [1280, 720]),
        ("t2i_014", "Noi_That", ["ĐỊNH NGHĨA LAI SỰ THƯ GIÃN AT PHÒNG KHÁCH", "SOFA CHỈNH ĐIỆN SMART ZEN", "TẶNG BÀN TRÀ MẶT ĐÁ CAO CẤP"], [1024, 768]),
        ("t2i_015", "FB_Kombucha", ["VÒNG EO THON GỌN VỚI CƠ THỂ NHẸ TÊNH SAU 14 NGÀY", "DETOX KOMBUCHA PREMIUM", "MUA LIỆU TRÌNH 14 NGÀY TẶNG BÌNH GIỮ NHIỆT"], [1024, 1280]),
        ("t2i_016", "Giao_Duc", ["ĐẬP TAN RÀO CẢN TIẾNG ANH TỰ TIN THĂNG TIẾN", "KHÓA HỌC GIAO TIẾP PHẢN XẠ ĐỘC QUYỀN", "TẶNG VOUCHER 1.000.000đ"], [1024, 1024]),
        ("t2i_017", "Gia_Dinh", ["TRẢ LAI KHÔNG GIAN SỐNG SẠCH BONG THƠM MÁT", "DEEP CLEANING HOME SERVICE", "GIẢM NGAY 20% CUỐI TUẦN"], [1024, 1024]),
        ("t2i_018", "Fintech", ["QUẢN LÝ CHI TIÊU THÔNG MINH TIỀN ĐẺ RA TIỀN", "WEALTHMASTER APP", "MIỄN PHÍ NÂNG CẤP PREMIUM 6 THÁNG"], [1024, 1280]),
        ("t2i_019", "Thiet_Bi_Bep", ["MÓN NƯỚNG NGOÀI GIÒN TRONG MỌNG NƯỚC MẸ NHÀN TÊNH", "NỒI CHIÊN HƠI NƯỚC CHEFPRO 15L", "TẶNG BỘ PHỤ KIỆN 5 MÓN"], [1280, 720]),
        ("t2i_020", "Studio_Cuoi", ["LƯU GIỮ KHOẢNH KHẮC THANH XUÂN RỰC RỠ NHẤT", "GÓI CHỤP ẢNH CƯỚI CINEMATIC LOVE", "TẶNG ẢNH CỔNG TRÁNG GƯƠNG PHA LÊ TRỊ GIÁ 3 TRIỆU"], [1024, 1536]),
        ("t2i_021", "Giac_Ngu", ["TẠM BIỆT ĐAU LƯNG NGỦ SÂU GIẤC ĐẾN SÁNG", "NỆM LÒ XO TÚI ĐỘC LẬP CLOUDSLEEP", "TRẢI NGHIỆM MIỄN PHÍ 100 ĐÊM"], [1024, 1024]),
    ]

    for idx in range(1, 51):
        case_id = f"t2i_{idx:03d}"
        if idx <= len(t2i_definitions):
            cid, cat, texts, size = t2i_definitions[idx-1]
            prompt = raw_prompts[idx-1] if idx-1 < len(raw_prompts) else f"Tạo ảnh quảng cáo poster thương mại cho sản phẩm {cat}. Văn bản bắt buộc: {', '.join(texts)}."
        else:
            cat = "Thoi_Trang" if idx % 2 == 0 else "FB_Thuc_Pham"
            texts = [f"KHUYẾN MÃI TẾT {idx}", f"GIẢM {idx*2}% CHỈ HÔM NAY"]
            size = [1024, 1024]
            prompt = f"Tạo banner quảng cáo phong cách Tết 2026 cho thương hiệu {cat}. Tiêu đề: \"{texts[0]}\". Ưu đãi: \"{texts[1]}\". Bố cục hiện đại, typography nét căng sắc nét."
        
        t2i_cases.append({
            "case_id": case_id,
            "track": "t2i",
            "reference_image": None,
            "instruction": prompt,
            "required_text": texts,
            "product_attributes": {
                "category": cat,
                "color": "default",
                "preserve_logo": False,
                "preserve_shape": False
            },
            "edit_type": None,
            "target_layout": "title_top_cta_bottom",
            "difficulty": "medium" if len(texts) <= 2 else "hard",
            "output_size": size
        })

    with open("benchmarks/tendoo_v0/cases/t2i.jsonl", "w", encoding="utf-8") as f:
        for item in t2i_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã tạo t2i.jsonl với {len(t2i_cases)} cases!")

    # ── 2. TẠO I2I.JSONL (50 CASES) ──
    i2i_cases = []
    edit_types = [
        "background_replacement",
        "lifestyle_placement",
        "key_visual",
        "preserve_packaging_logo",
        "object_removal_cleanup"
    ]

    for idx in range(1, 51):
        case_id = f"i2i_{idx:03d}"
        edit_t = edit_types[(idx - 1) % len(edit_types)]
        ref_cat = "simple_product" if idx % 3 == 1 else ("packaging_and_logo" if idx % 3 == 2 else "product_in_use")
        ref_path = f"references/{ref_cat}/prod_{idx:03d}.png"
        
        texts = [f"MUA NGAY - GIÁ {idx*10}.000đ", f"ƯU ĐÃI KHỦNG {idx}%"]
        
        i2i_cases.append({
            "case_id": case_id,
            "track": "i2i",
            "reference_image": ref_path,
            "instruction": f"Tạo ảnh quảng cáo từ sản phẩm reference {ref_path}. Yêu cầu {edit_t}: Đặt sản phẩm vào bối cảnh sang trọng chuyên nghiệp. Chèn văn bản bắt buộc: \"{texts[0]}\" và \"{texts[1]}\". Giữ nguyên logo và hình dáng sản phẩm.",
            "required_text": texts,
            "product_attributes": {
                "category": f"category_{idx}",
                "color": "original",
                "preserve_logo": True,
                "preserve_shape": True
            },
            "edit_type": edit_t,
            "target_layout": "product_center_text_overlay",
            "difficulty": "medium" if idx % 2 == 0 else "hard",
            "output_size": [1024, 1024]
        })

    with open("benchmarks/tendoo_v0/cases/i2i.jsonl", "w", encoding="utf-8") as f:
        for item in i2i_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã tạo i2i.jsonl với {len(i2i_cases)} cases!")

if __name__ == "__main__":
    generate_cases()
