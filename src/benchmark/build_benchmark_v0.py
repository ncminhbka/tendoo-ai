"""
Script xây dựng tập Test Set Benchmark v0 (50 Samples) cho TendooBizEval-Vi
kết hợp giữa Dữ liệu Thực tế Tendoo CSV và Prompts Chuẩn hóa.
"""

import json
import os
import csv
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def build_benchmark_v0():
    os.makedirs("data", exist_ok=True)
    os.makedirs("src/benchmark", exist_ok=True)

    # 1. Đọc dữ liệu mẫu từ CSV
    csv_file = "Untitled Discover session.csv"
    real_products = []
    
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "").strip()
                img_url = row.get("image", "").strip()
                price = row.get("price.current", "").strip()
                brand = row.get("brand", "No Brand").strip()
                if name and img_url and img_url.startswith("http"):
                    real_products.append({
                        "name": name,
                        "image_url": img_url,
                        "price": price,
                        "brand": brand if brand != "No Brand" else ""
                    })
                if len(real_products) >= 100:
                    break

    # 2. Xây dựng 50 Test Samples đa dạng ngành hàng
    samples = []
    
    # ── nhóm 1: TEXT-TO-IMAGE (25 Samples) ──
    t2i_configs = [
        {
            "domain": "Dong_Ho_Thoi_Trang", "format": "Poster_Instagram", "aspect_ratio": "4:5",
            "prompt": "Một chiếc đồng hồ thông minh hiện đại cao cấp với dây đeo kim loại màu bạc bóng bẩy, đặt trên bàn gỗ mộc mạc cạnh tách cà phê latte art. Ánh nắng ban mai dịu nhẹ. Phía trên cùng có văn bản \"THỜI GIAN LÀ CỦA BẠN\" phông chữ sans-serif trắng. Phía dưới có văn bản \"NÂNG TẦM PHONG CÁCH ĐỜI SỐNG\" màu trắng tinh tế. 8k, realistic.",
            "required_texts": ["THỜI GIAN LÀ CỦA BẠN", "NÂNG TẦM PHONG CÁCH ĐỜI SỐNG"],
            "difficulty": "Medium"
        },
        {
            "domain": "Fitness_Gym", "format": "Poster_Facebook", "aspect_ratio": "1:1",
            "prompt": "Tạo ảnh quảng cáo feedback khách hàng cho phòng tập Gym & PT cao cấp. Nền phòng gym sang trọng tone đen đỏ kịch tính. Tiêu đề phía trên: \"KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY THAY ĐỔI?\". Tên dịch vụ ở giữa: \"PRIVATE COACHING 1:1\". Ưu đãi góc dưới: \"GIẢM 20% GÓI PT THÁNG ĐẦU\". Phông chữ thể thao đậm, nổi bật.",
            "required_texts": ["KHÁCH HÀNG NÓI GÌ SAU 90 NGÀY THAY ĐỔI?", "PRIVATE COACHING 1:1", "GIẢM 20% GÓI PT THÁNG ĐẦU"],
            "difficulty": "Hard"
        },
        {
            "domain": "Spa_Thu_Cung", "format": "Social_Post", "aspect_ratio": "1:1",
            "prompt": "Ảnh quảng cáo dịch vụ Spa Thú cưng cao cấp. Background tiệm spa tone màu pastel hồng mint sáng rạng rỡ. Hình ảnh chú chó Poodle siêu đáng yêu sau khi cắt tỉa. Tiêu đề phía trên: \"BOSS LỘT XÁC THẾ NÀO SAU 2 GIỜ?\". Tên gói dịch vụ ở giữa: \"PREMIUM PET SPA\". Ưu đãi ở góc: \"TẶNG GÓI NGÂM SỤC OZON 200K\". Typography bo tròn thân thiện.",
            "required_texts": ["BOSS LỘT XÁC THẾ NÀO SAU 2 GIỜ?", "PREMIUM PET SPA", "TẶNG GÓI NGÂM SỤC OZON 200K"],
            "difficulty": "Hard"
        },
        {
            "domain": "Du_Lich_Glamping", "format": "Poster_Banner", "aspect_ratio": "16:9",
            "prompt": "Ảnh quảng cáo khu cắm trại lều glamping sang trọng giữa rừng thông hoàng hôn. Ánh đèn lều vàng ấm áp. Tiêu đề chính lớn ở giữa: \"TRẢI NGHIỆM CHỮA LÀNH GIỮA THIÊN NHIÊN\". Tên khu nghỉ dưỡng: \"CLOUD RETREAT GLAMPING\". Khuyến mãi góc dưới: \"GIẢM 30% ĐẶT PHÒNG SỚM\". Tone màu cam ấm thanh lịch.",
            "required_texts": ["TRẢI NGHIỆM CHỮA LÀNH GIỮA THIÊN NHIÊN", "CLOUD RETREAT GLAMPING", "GIẢM 30% ĐẶT PHÒNG SỚM"],
            "difficulty": "Medium"
        },
        {
            "domain": "Noi_That", "format": "Poster_Ad", "aspect_ratio": "4:3",
            "prompt": "Ảnh quảng cáo sofa thông minh nội thất cao cấp. Background phòng khách penthouse view thành phố về đêm lung linh. Ghế sofa da bò Ý màu nâu sang trọng. Tiêu đề phía trên: \"ĐỊNH NGHĨA LAI SỰ THƯ GIÃN\". Tên sản phẩm: \"SOFA CHỈNH ĐIỆN SMART ZEN\". Ưu đãi góc dưới: \"TẶNG BÀN TRÀ MẶT ĐÁ CAO CẤP\". Ánh sáng studio êm dịu.",
            "required_texts": ["ĐỊNH NGHĨA LAI SỰ THƯ GIÃN", "SOFA CHỈNH ĐIỆN SMART ZEN", "TẶNG BÀN TRÀ MẶT ĐÁ CAO CẤP"],
            "difficulty": "Medium"
        },
        {
            "domain": "FB_Kombucha", "format": "Poster_Instagram", "aspect_ratio": "4:5",
            "prompt": "Ảnh quảng cáo thức uống Kombucha trái cây tươi mát. Background lá bạc hà và trái cây tươi với hiệu ứng nước văng sinh động. Chai Kombucha ngập tràn sức sống. Tiêu đề phía trên: \"VÒNG EO THON GỌN SAU 14 NGÀY\". Tên sản phẩm: \"DETOX KOMBUCHA PREMIUM\". Khuyến mãi phía dưới: \"MUA 14 NGÀY TẶNG BÌNH GIỮ NHIỆT\". Tone màu xanh lá cam tươi tắn.",
            "required_texts": ["VÒNG EO THON GỌN SAU 14 NGÀY", "DETOX KOMBUCHA PREMIUM", "MUA 14 NGÀY TẶNG BÌNH GIỮ NHIỆT"],
            "difficulty": "Hard"
        },
        {
            "domain": "Giao_Duc_Ngoai_Ngu", "format": "Poster_Facebook", "aspect_ratio": "1:1",
            "prompt": "Ảnh quảng cáo khóa học tiếng Anh giao tiếp công sở. Môi trường văn phòng quốc tế hiện đại. Hình ảnh học viên tự tin thuyết trình. Tiêu đề chính phía trên: \"ĐẬP TAN RÀO CẢN TIẾNG ANH\". Tên khóa học: \"GIAO TIẾP PHẢN XẠ 1 KÈM 1\". Ưu đãi góc dưới: \"TẶNG VOUCHER 1.000.000đ\". Tone màu xanh navy vàng gold uy tín.",
            "required_texts": ["ĐẬP TAN RÀO CẢN TIẾNG ANH", "GIAO TIẾP PHẢN XẠ 1 KÈM 1", "TẶNG VOUCHER 1.000.000đ"],
            "difficulty": "Hard"
        },
        {
            "domain": "Thiet_Bi_Bep", "format": "Poster_Ad", "aspect_ratio": "16:9",
            "prompt": "Ảnh quảng cáo nồi chiên hơi nước ChefPro cao cấp. Background căn bếp hiện đại. Con gà quay vàng ươm khói bốc nghi ngút bên trong nồi chiên. Tiêu đề phía trên: \"MÓN NƯỚNG NGOÀI GIÒN TRONG MỌNG NƯỚC\". Tên sản phẩm: \"NỒI CHIÊN HƠI NƯỚC CHEFPRO 15L\". Ưu đãi: \"TẶNG BỘ PHỤ KIỆN 5 MÓN\". Tone đen vàng ấm áp.",
            "required_texts": ["MÓN NƯỚNG NGOÀI GIÒN TRONG MỌNG NƯỚC", "NỒI CHIÊN HƠI NƯỚC CHEFPRO 15L", "TẶNG BỘ PHỤ KIỆN 5 MÓN"],
            "difficulty": "Medium"
        },
        {
            "domain": "Studio_Cuoi", "format": "Poster_Instagram", "aspect_ratio": "2:3",
            "prompt": "Ảnh quảng cáo dịch vụ studio cưới phong cách cinematic lãng mạn. Background bình minh trên bãi biển ngập nắng vàng. Cô dâu chú rể trao nhau nụ hôn hạnh phúc. Tiêu đề bay bổng phía trên: \"LƯU GIỮ KHOẢNH KHẮC THANH XUÂN\". Tên gói: \"CINEMATIC LOVE WEDDING\". Ưu đãi góc dưới: \"TẶNG ẢNH CỔNG TRÁNG GƯƠNG PHA LÊ\". Tone màu be ánh sáng vàng ấm.",
            "required_texts": ["LƯU GIỮ KHOẢNH KHẮC THANH XUÂN", "CINEMATIC LOVE WEDDING", "TẶNG ẢNH CỔNG TRÁNG GƯƠNG PHA LÊ"],
            "difficulty": "Medium"
        },
        {
            "domain": "My_Pham_Skincare", "format": "Poster_Ad", "aspect_ratio": "4:5",
            "prompt": "Ảnh quảng cáo kem dưỡng da chống lão hóa cao cấp. Background hiệu ứng giọt sương và cánh hoa hồng mềm mại. Tiêu đề chính phía trên: \"TÁI TẠO LÀN DA CĂNG BÓNG\". Tên sản phẩm: \"SERUM PEPTIDE REPAIR\". Giá ưu đãi ở dưới: \"GIÁ CHỈ 490.000đ (GIẢM 35%)\". Tone màu hồng pastel sang trọng.",
            "required_texts": ["TÁI TẠO LÀN DA CĂNG BÓNG", "SERUM PEPTIDE REPAIR", "GIÁ CHỈ 490.000đ (GIẢM 35%)"],
            "difficulty": "Hard"
        }
    ]

    # Nhân bản các dạng t2i để tạo đủ 25 mẫu t2i đa dạng
    idx = 1
    for cfg in t2i_configs:
        samples.append({
            "id": f"tendoo_t2i_{idx:03d}",
            "task_type": "text_to_image",
            "domain": cfg["domain"],
            "format": cfg["format"],
            "aspect_ratio": cfg["aspect_ratio"],
            "prompt": cfg["prompt"],
            "reference_image_url": None,
            "required_texts": cfg["required_texts"],
            "layout_requirements": "Text tiêu đề ở 1/3 phía trên, tên sản phẩm ở giữa, ưu đãi/CTA ở góc dưới cùng. Không đè chữ lên vật thể chính.",
            "visual_requirements": "Ảnh thương mại sắc nét, độ phân giải cao, bố cục quảng cáo chuyên nghiệp.",
            "brand_attributes": cfg["domain"],
            "difficulty": cfg["difficulty"],
            "expected_language": "vi"
        })
        idx += 1

    # Thêm thêm các case t2i cho đủ 25
    extra_domains = [
        ("Nong_San_Viet", "TÁI TẠO NĂNG LƯỢNG MỖI NGÀY", "TRÀ SÂM ĐINH LĂNG VIỆT", "ƯU ĐÃI MUA 2 TẶNG 1"),
        ("Tra_Sua_An_Vat", "ĐẬM VỊ TRÀ THƠM VỊ SỮA", "TRÀ SỮA Ô LONG NƯỚNG", "GIẢM 20% ĐƠN ĐẦU TÊN"),
        ("Tiem_Banh_Bakery", "HƯƠNG VỊ NGỌT NGÀO TỪ TÂM", "BÁNH KEM SINH NHẬT ART", "FREESHIP NỘI THÀNH 5KM"),
        ("Dich_Vu_Ve_Sinh", "TRẢ LAI KHÔNG GIANG SẠCH BONG", "DEEP CLEANING HOME", "GIẢM 15% CUỐI TUẦN"),
        ("Bao_Hiem_Tai_Chinh", "AN TÂM TƯƠNG LAI GIA ĐÌNH", "BẢO HIỂM SỨC KHỎE TOÀN DIỆN", "CHỈ TỪ 15.000đ/NGÀY"),
    ]
    for dom, t1, t2, t3 in extra_domains:
        if idx > 25: break
        samples.append({
            "id": f"tendoo_t2i_{idx:03d}",
            "task_type": "text_to_image",
            "domain": dom,
            "format": "Poster_Ad",
            "aspect_ratio": "1:1",
            "prompt": f"Ảnh quảng cáo cho thương hiệu {dom}. Phía trên có tiêu đề \"{t1}\". Ở giữa là \"{t2}\". Phía dưới có khuyến mãi \"{t3}\". Ánh sáng sắc nét, typography nổi bật.",
            "reference_image_url": None,
            "required_texts": [t1, t2, t3],
            "layout_requirements": "Phân cấp tiêu đề rõ ràng, màu chữ tương phản dễ đọc.",
            "visual_requirements": "Chất lượng thương mại cao cấp.",
            "brand_attributes": dom,
            "difficulty": "Medium",
            "expected_language": "vi"
        })
        idx += 1

    # ── nhóm 2: PRODUCT-TO-BANNER (REFERENCE IMAGE) (15 Samples) ──
    ref_idx = 1
    for p in real_products[:15]:
        p_name = p["name"]
        p_price = p["price"]
        p_url = p["image_url"]
        p_brand = p["brand"]
        
        samples.append({
            "id": f"tendoo_ref_{ref_idx:03d}",
            "task_type": "product_to_banner",
            "domain": "Retail_SME_Tendoo",
            "format": "Product_Placement_Ad",
            "aspect_ratio": "1:1",
            "prompt": f"Đặt sản phẩm từ ảnh reference [{p_name}] vào bối cảnh poster quảng cáo sang trọng. Phía trên chèn chữ tiêu đề \"SẢN PHẨM CHÍNH HÃNG\". Phía dưới chèn giá bán \"GIÁ CHỈ {p_price}đ\". Bố cục hài hòa, nổi bật sản phẩm.",
            "reference_image_url": p_url,
            "required_texts": ["SẢN PHẨM CHÍNH HÃNG", f"GIÁ CHỈ {p_price}đ"],
            "layout_requirements": "Giữ nguyên chi tiết sản phẩm từ ảnh reference ở trung tâm, tiêu đề chữ ở phía trên, giá tiền ở góc dưới.",
            "visual_requirements": "Bối cảnh sân khấu quảng cáo studio chuyên nghiệp.",
            "brand_attributes": p_brand if p_brand else "Tendoo Seller",
            "difficulty": "Hard",
            "expected_language": "vi"
        })
        ref_idx += 1

    # ── nhóm 3: IMAGE EDITING / INPAINTING (10 Samples) ──
    edit_idx = 1
    for p in real_products[15:25]:
        p_name = p["name"]
        p_url = p["image_url"]
        
        samples.append({
            "id": f"tendoo_edit_{edit_idx:03d}",
            "task_type": "image_editing",
            "domain": "Image_Inpainting_Edit",
            "format": "Banner_Editing",
            "aspect_ratio": "1:1",
            "prompt": f"Chỉnh sửa ảnh banner gốc [{p_name}]: Giữ nguyên sản phẩm chính, thay thế vùng nền xung quanh thành phong cách Tết Việt Nam ngập sắc hoa đào xuân. Thêm dải banner đỏ phía trên với dòng chữ \"XUÂN PHÁT TÀI 2026\".",
            "reference_image_url": p_url,
            "required_texts": ["XUÂN PHÁT TÀI 2026"],
            "layout_requirements": "Giữ nguyên vật thể sản phẩm gốc ở chính giữa, chỉ thay đổi background và vẽ thêm dải chữ Tết phía trên.",
            "visual_requirements": "Màu đỏ may mắn phong cách Tết Việt Nam, hoa đào nở rộn ràng.",
            "brand_attributes": "Tendoo Merchant",
            "difficulty": "Hard",
            "expected_language": "vi"
        })
        edit_idx += 1

    # 3. Xuất tệp JSON chuẩn
    output_json = "data/tendoo_biz_eval_v0.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã khởi tạo thành công tập Benchmark `{output_json}` với {len(samples)} Test Samples!")

if __name__ == "__main__":
    build_benchmark_v0()
