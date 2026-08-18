"""
TendooBizEval-Vi Benchmark v2 Suite Builder
Tự động khởi tạo tập test cases T2I và I2I bám sát nghiệp vụ thực tế của SME Tendoo AI:
- Poster Khuyến Mại & Ưu Đãi (Sale 50%, Giờ vàng, Mua 1 tặng 1)
- Ảnh Giới Thiệu Sản Phẩm (Skincare, Mỹ phẩm, Đồ điện tử, Văn phòng phẩm)
- Banner Khai Trương (Grand Opening, Mừng khai trương chi nhánh)
- Ảnh Feedback Khách Hàng (Góc feedback, Cảm ơn quý khách)
- Ảnh Tin Tuyển Dụng (Tuyển nhân viên bán hàng, pha chế)
- Menu Món Ăn & Đồ Uống (Thực đơn Phở Bò, Menu trà trái cây)
- Yêu Cầu Sinh Tự Do & Hiểu Văn Hóa Việt (Tết 2026, Áo dài, Nón lá, Hoa đào)
- STRESS TEST 1: Dấu Tiếng Việt Khó & Phức Tạp (ẫ, ỡ, ẻ, ửng, nghễu...)
- STRESS TEST 2: Số, Định Dạng Tiền Tệ (199.000đ, 500.000Đ) & SĐT/Hotline (0987.654.321, 1900-8198)
- STRESS TEST 3: Text Đa Dòng & Đa Cấp Độ Cỡ Chữ (Tiêu đề + Subtitle + Giá + Hotline)
- Chỉnh Sửa Ảnh Tự Do I2I (Thay nền studio, Ghép sản phẩm lên bục đá, Bàn gỗ)
"""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "benchmarks" / "tendoo_v1" / "manifests" / "reference_manifest.json"
CASES_DIR = ROOT / "benchmarks" / "tendoo_v1" / "cases"

FLUX2_RESOLUTIONS = [
    [1024, 1024],  # 1:1
    [1024, 1280],  # 4:5
    [1280, 720],   # 16:9
    [1024, 1536],  # 2:3
    [1024, 768],   # 4:3
    [1088, 1920],  # 9:16
    [1200, 624]    # 1.91:1
]

LAYOUTS = ["square_1x1", "vertical_4x5", "horizontal_16x9", "vertical_2x3", "horizontal_4x3", "story_9x16"]
TEXT_LENGTHS = ["short", "medium", "long"]
DIFFICULTIES = ["easy", "medium", "hard"]
EDIT_TYPES = [
    "background_replacement",
    "lifestyle_placement",
    "key_visual",
    "preserve_packaging_logo",
    "object_removal_cleanup"
]

# Danh sách mẫu T2I bám sát nhu cầu thực tế người dùng SME Tendoo + Stress test tiếng Việt nâng cao
T2I_DATA = [
    # 1. Poster Khuyến Mại & Ưu Đãi
    ("Poster khuyến mại giảm 50% chương trình Giờ Vàng Giá Sốc với tiêu đề SIÊU SALE MÙA HÈ gam màu đỏ rực rỡ", ["SIÊU SALE MÙA HÈ"]),
    ("Poster ưu đãi MUA 1 TẶNG 1 dành cho quán trà sữa kèm chữ ƯU ĐÃI TUẦN VÀNG phông chữ bo tròn xinh xắn", ["MUA 1 TẶNG 1", "ƯU ĐÃI TUẦN VÀNG"]),
    ("Poster quảng cáo chương trình tri ân khách hàng giảm 30% toàn bộ gian hàng thời trang tiêu đề TRI ÂN KHÁCH HÀNG", ["TRI ÂN KHÁCH HÀNG"]),
    ("Poster flash sale giảm sâu shop mỹ phẩm với tiêu đề FLASH SALE 70% ánh sáng neon lấp lánh náo nhiệt", ["FLASH SALE 70%"]),
    ("Banner ưu đãi cuối tuần siêu thị gia dụng tiêu đề WEEKEND SALE GIÁ SỐC MỖI NGÀY màu xanh mát", ["WEEKEND SALE GIÁ SỐC MỖI NGÀY"]),
    ("Poster khuyến mại cho shop quần áo nữ phông nền pastel thanh lịch kèm chữ ĐỒNG GIÁ 99K XẢ KHO CỰC ĐẠI", ["ĐỒNG GIÁ 99K", "XẢ KHO CỰC ĐẠI"]),
    ("Banner tặng voucher cho đơn hàng đầu tiên ứng dụng Tendoo với tiêu đề TẶNG VOUCHER 50K hiện đại", ["TẶNG VOUCHER 50K"]),
    ("Poster khuyến mại sinh nhật shop 5 tuổi với tiêu đề CHÚC MỪNG SINH NHẬT GIẢM GIÁ 40% pháo hoa rực rỡ", ["CHÚC MỪNG SINH NHẬT GIẢM GIÁ 40%"]),

    # 2. Ảnh Giới Thiệu Sản Phẩm
    ("Tạo ảnh quảng cáo Chai kem chống nắng đặt trên bục gỗ giữa không gian nắng ấm thiên nhiên mùa hè tươi mát", ["Chai kem chống nắng"]),
    ("Hình ảnh giới thiệu Bộ sản phẩm skincare sang trọng trên khay đá cẩm thạch trắng tiêu đề SERUM DƯỠNG DA VÀNG", ["Bộ sản phẩm skincare", "SERUM DƯỠNG DA VÀNG"]),
    ("Banner giới thiệu Nước hoa cao cấp với ánh sáng nghệ thuật studio chữ nổi bật LUXE PERFUME QUYẾN RŨ", ["Nước hoa cao cấp", "LUXE PERFUME QUYẾN RŨ"]),
    ("Quảng cáo Tai nghe chụp tai không dây màu đen matte hiện đại tiêu đề ÂM THANH CHÂN THỰC CHỐNG ỒN CỰC ĐỈNH", ["Tai nghe chụp tai không dây", "ÂM THANH CHÂN THỰC CHỐNG ỒN CỰC ĐỈNH"]),
    ("Giới thiệu Đồng hồ nam dây da Thụy Sĩ cổ điển đặt trên bề mặt da sang trọng dòng chữ ĐẲNG CẤP THỜI GIAN", ["Đồng hồ nam dây da Thụy Sĩ", "ĐẲNG CẤP THỜI GIAN"]),
    ("Ảnh chụp studio quảng cáo Đôi giày sneaker thể thao màu trắng tinh khôi tiêu đề THỜI TRANG NĂNG ĐỘNG 2026", ["Đôi giày sneaker thể thao", "THỜI TRANG NĂNG ĐỘNG 2026"]),
    ("Poster giới thiệu Tay cầm chơi game màu đỏ phong cách gaming esports tiêu đề CHIẾN GAME CỰC ĐỈNH", ["Tay cầm chơi game màu đỏ", "CHIẾN GAME CỰC ĐỈNH"]),
    ("Hình ảnh giới thiệu Bút chì màu đủ các loại màu sắp xếp nghệ thuật dải cầu vồng trên bàn gỗ sáng tạo", ["Bút chì màu đủ các loại màu"]),

    # 3. Banner Khai Trương & Sự Kiện
    ("Banner mừng khai trương chi nhánh mới với tiêu đề MỪNG KHAI TRƯƠNG giảm 20% pháo hoa rực rỡ may mắn", ["MỪNG KHAI TRƯƠNG"]),
    ("Poster sự kiện GRAND OPENING QUÁN PHỞ GIA TRUYỀN hoa chúc mừng hai bên sang trọng rực rỡ đỏ may mắn", ["GRAND OPENING QUÁN PHỞ GIA TRUYỀN"]),
    ("Banner khai trương cửa hàng thời trang trẻ em tiêu đề KHAI TRƯƠNG RỘN RÀNG NHẬN QUÀ BẬT TÔNG tươi vui", ["KHAI TRƯƠNG RỘN RÀNG NHẬN QUÀ BẬT TÔNG"]),
    ("Poster thông báo sự kiện ra mắt bộ sưu tập thu đông mới tiêu đề RA MẮT BỘ SỰ TẬP MỚI 2026 tinh tế", ["RA MẮT BỘ SỰ TẬP MỚI 2026"]),
    ("Banner mở bán căn hộ mẫu dự án nhà ở tiêu đề MỞ BÁN CHÍNH THỨC NHẬN NHÀ NGAY đẳng cấp sang trọng", ["MỞ BÁN CHÍNH THỨC NHẬN NHÀ NGAY"]),
    ("Poster lễ khai trương tiệm nướng BBQ Hàn Quốc tiêu đề WELCOME TO BBQ HOUSE GIẢM 15% TỔNG HÓA ĐƠN thơm lừng", ["WELCOME TO BBQ HOUSE GIẢM 15% TỔNG HÓA ĐƠN"]),
    ("Banner sự kiện đêm nhạc hội mừng thành lập công ty tiêu đề ĐÊM NHẠC TRI ÂN KẾT NỐI VỚI TENDOO hoành tráng", ["ĐÊM NHẠC TRI ÂN KẾT NỐI VỚI TENDOO"]),

    # 4. Ảnh Feedback Khách Hàng
    ("Ảnh khung feedback khách hàng tiêu đề GÓC FEEDBACK KHÁCH YÊU đính kèm đánh giá 5 sao CẢM ƠN QUÝ KHÁCH", ["GÓC FEEDBACK KHÁCH YÊU", "CẢM ƠN QUÝ KHÁCH"]),
    ("Banner góc cảm nhận khách hàng REVIEW THỰC TẾ SẢN PHẨM gam màu xanh lá organic dịu mát uy tín", ["REVIEW THỰC TẾ SẢN PHẨM"]),
    ("Ảnh feedback khách khen ngợi son môi mịn màng tiêu đề KHÁCH HÀNG NÓI GÌ VỀ CHÚNG TÔI chân thực", ["KHÁCH HÀNG NÓI GÌ VỀ CHÚNG TÔI"]),
    ("Poster tổng hợp đánh giá 5 sao từ khách hàng mua giày thể thao dòng chữ 100% KHÁCH HÀNG HÀI LÒNG", ["100% KHÁCH HÀNG HÀI LÒNG"]),
    ("Banner góc cảm ơn ủng hộ shop 10000 đơn hàng tiêu đề 10000 LỜI CẢM ƠN CHÂN THÀNH tri ân sâu sắc", ["10000 LỜI CẢM ƠN CHÂN THÀNH"]),
    ("Ảnh khung feedback trải nghiệm Spa chăm sóc da mặt dòng chữ LÀN DA CĂNG BÓNG SAU 1 LIỆU TRÌNH tươi trẻ", ["LÀN DA CĂNG BÓNG SAU 1 LIỆU TRÌNH"]),
    ("Poster feedback khách hàng khen Phở bò đậm vị gia truyền tiêu đề ĐÁNH GIÁ THẬT TỪ THỰC KHÁCH hấp dẫn", ["ĐÁNH GIÁ THẬT TỪ THỰC KHÁCH"]),

    # 5. Ảnh Tin Tuyển Dụng
    ("Poster tuyển dụng nhân viên bán hàng thời trang tiêu đề THÔNG BÁO TUYỂN DỤNG ghi LƯƠNG HẤP DẪN MÔI TRƯỜNG NĂNG ĐỘNG", ["THÔNG BÁO TUYỂN DỤNG", "LƯƠNG HẤP DẪN MÔI TRƯỜNG NĂNG ĐỘNG"]),
    ("Banner tìm nhân viên pha chế trà sữa & thu ngân tiêu đề TUYỂN DỤNG ĐỒNG ĐỘI màu cam năng động thu hút", ["TUYỂN DỤNG ĐỒNG ĐỘI"]),
    ("Poster tuyển dụng vị trí Nhân viên Marketing Online cho shop dòng chữ GIA NHẬP ĐỘI NGŨ TENDOO MEDIA", ["GIA NHẬP ĐỘI NGŨ TENDOO MEDIA"]),
    ("Banner tuyển dụng nhân viên giao hàng shipper tiêu đề TUYỂN SHIPPER THU NHẬP CAO CHẠY NGAY nhanh chóng", ["TUYỂN SHIPPER THU NHẬP CAO CHẠY NGAY"]),
    ("Poster tuyển dụng đầu bếp nấu món Việt nhà hàng tiêu đề TUYỂN ĐẦU BẾP MÓN VIỆT CHẾ ĐỘ ĐÃI NGỘ TỐT", ["TUYỂN ĐẦU BẾP MÓN VIỆT CHẾ ĐỘ ĐÃI NGỘ TỐT"]),
    ("Banner thông báo tuyển thực tập sinh thiết kế đồ họa dòng chữ TUYỂN DESIGNER INTERN THỎAI MÁI SÁNG TẠO", ["TUYỂN DESIGNER INTERN THỎAI MÁI SÁNG TẠO"]),
    ("Poster tuyển dụng chuyên viên tư vấn Spa tiêu đề TUYỂN CHUYÊN VIÊN TƯ VẤN SPA THU NHẬP 15-20 TRIỆU", ["TUYỂN CHUYÊN VIÊN TƯ VẤN SPA THU NHẬP 15-20 TRIỆU"]),

    # 6. Menu Món Ăn & Đồ Uống
    ("Thiết kế menu quán Phở Bò Hà Nội tiêu đề THỰC ĐƠN GIA TRUYỀN liệt kê Phở Tái Phở Nạm Phở Bắp màu nâu gỗ cổ truyền", ["THỰC ĐƠN GIA TRUYỀN", "Phở Tái", "Phở Nạm", "Phở Bắp"]),
    ("Poster menu đồ uống mùa hè MENU TRÀ TRÁI CÂY TƯƠI kèm bảng giá mát lạnh và hình ảnh ly nước giải nhiệt", ["MENU TRÀ TRÁI CÂY TƯƠI"]),
    ("Thiết kế bảng menu bánh ngọt & cà phê tiệm bakery tiêu đề MENU CAFE BAKERY phong cách Vintage ấm cúng", ["MENU CAFE BAKERY"]),
    ("Poster menu combo bữa sáng tiết kiệm COMBO BỮA SÁNG 35K gồm 1 Bánh mì thịt nướng và 1 Tách Cà phê phin đậm đà", ["COMBO BỮA SÁNG 35K", "Bánh mì thịt nướng", "Tách Cà phê phin"]),
    ("Thiết kế menu lẩu nướng buffet hải sản tiêu đề MENU BUFFET HẢI SẢN 199K trình bày món ăn hấp dẫn tươi ngon", ["MENU BUFFET HẢI SẢN 199K"]),
    ("Poster thực đơn trà sữa trân châu đường đen & bingsu hoa quả dòng chữ BEST SELLER DRINKS 2026 hiện đại", ["BEST SELLER DRINKS 2026"]),

    # 7. Yêu Cầu Sinh Tự Do & Văn Hóa Việt
    ("Poster Tết cổ truyền Việt Nam 2026 Bính Ngọ hoa đào nở rộ câu đối đỏ CHÚC MỪNG NĂM MỚI em bé mặc áo dài gấm đỏ", ["CHÚC MỪNG NĂM MỚI"]),
    ("Tạo ảnh nghệ thuật cô gái Việt Nam mặc Áo dài trắng đội Nón lá sen đứng bên hồ sen Hà Nội buổi sáng sớm trong vắt dòng chữ NÉT ĐẸP VIỆT NAM", ["NÉT ĐẸP VIỆT NAM", "Áo dài trắng", "Nón lá sen"]),
    ("Poster quảng cáo mâm cơm Tết Việt Nam ấm cúng gia đình với dưa hành bánh chưng xanh giò lụa câu đối TẾT DOANH VIÊN", ["TẾT DOANH VIÊN", "bánh chưng xanh"]),
    ("Tạo ảnh quảng cáo phố cổ Hà Nội mùa thu vắng lặng với gánh hàng hoa cúc họa mi vàng rực rỡ dòng chữ HÀ NỘI MÙA THU", ["HÀ NỘI MÙA THU"]),
    ("Poster nghệ thuật đêm hội Trung Thu Việt Nam với đèn ông sao múa lân sôi động dòng chữ TẾT TRUNG THU SUM VẬY", ["TẾT TRUNG THU SUM VẬY"]),
    ("Tạo ảnh minh họa cảnh làm gốm sứ Bát Tràng truyền thống với bàn xoay đất sét hoa văn men ngọc TINH HOA GỐM VIỆT", ["TINH HOA GỐM VIỆT"]),
    ("Poster phong cách Pop-Art hiện đại chào mừng Quốc Khánh 2/9 cờ đỏ sao vàng khẩu hiệu TỰ HÀO VIỆT NAM hào hùng", ["TỰ HÀO VIỆT NAM"]),

    # 8. STRESS TEST 1: Dấu Tiếng Việt Khó & Phức Tạp (diacritics_stress_test)
    ("Poster quảng cáo mỹ phẩm tự nhiên tiêu đề NGHỆ THUẬT NGHỄU NGHỆ VỚI LÀN DA RỰC RỠ ỬNG HỒNG phong cách spa dịu mát", ["NGHỆ THUẬT NGHỄU NGHỆ", "ỬNG HỒNG"]),
    ("Banner giới thiệu món ăn bổ dưỡng tiêu đề BỔ DƯỠNG NGHỆ HOÀNG VÀ TỔ YẾN NGHỄU NGHỆ màu vàng kim sang trọng", ["BỔ DƯỠNG NGHỆ HOÀNG", "NGHỄU NGHỆ"]),
    ("Poster thông báo mở cửa shop phong cách Vintage tiêu đề CỬA HẢO BẠNG MỞ CỬA HỎA NỐC RỘN RÀNG", ["CỬA HẢO BẠNG", "MỞ CỬA"]),
    ("Banner quảng cáo serum dưỡng da dòng chữ NĂNG LƯỢNG RỰC RỠ ỬNG HỒNG CHO LÀN DA TƯƠI TRẺ", ["NĂNG LƯỢNG RỰC RỠ", "ỬNG HỒNG"]),
    ("Poster nghệ thuật ẩm thực truyền thống tiêu đề HƯƠNG VỊ RỰC RỠ ĐƠN DỰNG NGHỄU NGHỆ đậm đà", ["HƯƠNG VỊ RỰC RỠ", "NGHỄU NGHỆ"]),

    # 9. STRESS TEST 2: Số, Định Dạng Tiền Tệ (VND/đ) & Hotline Thực Tế (currency_phone_stress_test)
    ("Poster khuyến mại ưu đãi sốt dẻo tiệm trà sữa với tiêu đề ĐỒNG GIÁ 19.000đ kèm chữ Hotline: 0987.654.321", ["ĐỒNG GIÁ 19.000đ", "Hotline: 0987.654.321"]),
    ("Banner flash sale shop thời trang tiêu đề GIẢM NÓNG 500.000Đ DÀNH CHO 100 KHÁCH HÀNG ĐẦU TIÊN", ["GIẢM NÓNG 500.000Đ", "100 KHÁCH HÀNG"]),
    ("Poster menu combo bữa ăn gia đình tiêu đề COMBO GIA ĐÌNH CHỈ 299.000đ liên hệ Tổng đài: 1900-8198", ["COMBO GIA ĐÌNH CHỈ 299.000đ", "Tổng đài: 1900-8198"]),
    ("Banner khuyến mại mỹ phẩm chính hãng tiêu đề GIÁ CHỈ 199.000đ tư vấn ngay 0912.345.678", ["GIÁ CHỈ 199.000đ", "0912.345.678"]),
    ("Poster voucher xả kho điện máy tiêu đề VOUCHER 1.000.000đ HOTLINE DỊCH VỤ: 1800-1060", ["VOUCHER 1.000.000đ", "HOTLINE DỊCH VỤ: 1800-1060"]),

    # 10. STRESS TEST 3: Text Đa Dòng & Đa Cấp Độ Cỡ Chữ (multiline_hierarchy_stress_test)
    ("Banner quảng cáo siêu thị điện máy có tiêu đề chính SIÊU SALE MÙA HÈ, slogan nhỏ GIẢM GIÁ LÊN TỚI 50%, và giá GIÁ CHỈ 4.990.000đ kèm Hotline: 0987.654.321", ["SIÊU SALE MÙA HÈ", "GIẢM GIÁ LÊN TỚI 50%", "GIÁ CHỈ 4.990.000đ", "Hotline: 0987.654.321"]),
    ("Poster quảng cáo kem dưỡng da có tiêu đề lớn SERUM DƯỠNG TRẮNG, dòng phụ LÀN DA RỰC RỠ ỬNG HỒNG, giá ƯU ĐÃI 299.000đ và hotline 0909.123.456", ["SERUM DƯỠNG TRẮNG", "LÀN DA RỰC RỠ ỬNG HỒNG", "ƯU ĐÃI 299.000đ", "0909.123.456"]),
    ("Poster thực đơn bánh mì nóng có tiêu đề THỰC ĐƠN BÁNH MÌ NÓNG, dòng giá Ổ ĐẶC BIỆT 35.000đ, slogan MUA 3 TẶNG 1, hotline Tổng đài: 1900-8198", ["THỰC ĐƠN BÁNH MÌ NÓNG", "Ổ ĐẶC BIỆT 35.000đ", "MUA 3 TẶNG 1", "Tổng đài: 1900-8198"]),
    ("Banner khai trương tiệm trà trái cây có tiêu đề GRAND OPENING TRÀ TƯƠI, sub-heading GIẢM 30% TOÀN BỘ MENU, giá CHỈ TỪ 25.000đ, Hotline: 0977.888.999", ["GRAND OPENING TRÀ TƯƠI", "GIẢM 30% TOÀN BỘ MENU", "CHỈ TỪ 25.000đ", "Hotline: 0977.888.999"]),
    ("Poster tuyển dụng nhân sự có tiêu đề TUYỂN DỤNG NHÂN VIÊN BÁN HÀNG, dòng phụ LƯƠNG 10-15 TRIỆU, hotline liên hệ Hotline: 0933.555.777", ["TUYỂN DỤNG NHÂN VIÊN BÁN HÀNG", "LƯƠNG 10-15 TRIỆU", "Hotline: 0933.555.777"])
]

def build_benchmark_v1_cases():
    print("🚀 BẮT ĐẦU DỰNG TẬP BENCHMARK V2 (TENDOO V1)...")
    
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest tại {MANIFEST_PATH}")
        
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    ref_list = manifest_data.get("references", [])
    print(f"  📌 Nạp {len(ref_list)} ảnh reference sản phẩm thực tế từ manifest.")

    t2i_cases = []
    i2i_cases = []

    # 1. Dựng T2I Cases
    for idx, (instruction, required_text) in enumerate(T2I_DATA, start=1):
        size = FLUX2_RESOLUTIONS[(idx - 1) % len(FLUX2_RESOLUTIONS)]
        layout = LAYOUTS[(idx - 1) % len(LAYOUTS)]
        difficulty = DIFFICULTIES[(idx - 1) % len(DIFFICULTIES)]
        text_len = TEXT_LENGTHS[(idx - 1) % len(TEXT_LENGTHS)]

        t2i_cases.append({
            "case_id": f"t2i_{idx:03d}",
            "track": "t2i",
            "instruction": instruction,
            "required_text": required_text,
            "target_layout": layout,
            "difficulty": difficulty,
            "text_length": text_len,
            "output_size": size
        })

    # 2. Dựng I2I Cases (1-đối-1 khớp với chính xác số lượng ảnh reference sản phẩm)
    I2I_PROMPT_PATTERNS = [
        ("Thay toàn bộ phông nền phía sau sản phẩm {product_name} thành bối cảnh studio ánh sáng vàng dịu sang trọng kèm dòng chữ SẢN PHẨM CAO CẤP", ["SẢN PHẨM CAO CẤP"], "background_replacement"),
        ("Thay nền sau sản phẩm {product_name} thành bãi biển nắng vàng biển xanh mát lạnh kèm tiêu đề MÙA HÈ RỰC RỠ", ["MÙA HÈ RỰC RỠ"], "background_replacement"),
        ("Ghép sản phẩm {product_name} lên mặt bàn gỗ trong không gian gian bếp ấm cúng hiện đại kèm chữ TRẢI NGHIỆM THỰC TẾ", ["TRẢI NGHIỆM THỰC TẾ"], "lifestyle_placement"),
        ("Đặt sản phẩm {product_name} vào bối cảnh góc làm việc công sở tối giản đẹp mắt kèm dòng chữ KHÔNG GIAN SÁNG TẠO", ["KHÔNG GIAN SÁNG TẠO"], "lifestyle_placement"),
        ("Biến sản phẩm {product_name} thành hình ảnh quảng cáo Key Visual trên poster khuyến mại với chữ SIÊU SALE GIỜ VÀNG", ["SIÊU SALE GIỜ VÀNG"], "key_visual"),
        ("Thiết kế banner Key Visual cho sản phẩm {product_name} với hiệu ứng tia sáng neon hiện đại tiêu đề ĐẲNG CẤP TENDOO", ["ĐẲNG CẤP TENDOO"], "key_visual"),
        ("Giữ nguyên logo và kiểu dáng bao bì {product_name} trong ảnh tham chiếu ghép vào khung poster ưu đãi MUA 1 TẶNG 1", ["MUA 1 TẶNG 1"], "preserve_packaging_logo"),
        ("Bảo tồn chi tiết logo sản phẩm {product_name} đặt trên khay đá cẩm thạch sang trọng kèm dòng chữ TỰ HÀO THƯƠNG HIỆU VIỆT", ["TỰ HÀO THƯƠNG HIỆU VIỆT"], "preserve_packaging_logo"),
        ("Tách sản phẩm {product_name} khỏi các chi tiết thừa trong ảnh tham chiếu đặt lên bục gỗ tự nhiên kèm chữ 100% ORGANIC", ["100% ORGANIC"], "object_removal_cleanup"),
        ("Làm sạch nền và loại bỏ vật thể gây xao nhãng xung quanh {product_name} làm nổi bật sản phẩm kèm dòng chữ CHẤT LƯỢNG HÀNG ĐẦU", ["CHẤT LƯỢNG HÀNG ĐẦU"], "object_removal_cleanup")
    ]

    for idx, ref in enumerate(ref_list, start=1):
        product_name = ref.get("title", "sản phẩm")
        ref_path = ref.get("path", "")
        
        edit_type = EDIT_TYPES[(idx - 1) % len(EDIT_TYPES)]
        
        size = FLUX2_RESOLUTIONS[(idx - 1) % len(FLUX2_RESOLUTIONS)]
        layout = LAYOUTS[(idx - 1) % len(LAYOUTS)]
        difficulty = DIFFICULTIES[(idx - 1) % len(DIFFICULTIES)]
        text_len = TEXT_LENGTHS[(idx - 1) % len(TEXT_LENGTHS)]

        pattern_tuple = I2I_PROMPT_PATTERNS[(idx - 1) % len(I2I_PROMPT_PATTERNS)]
        instruction_template, req_text_tpl, _ = pattern_tuple
        
        instruction = instruction_template.format(product_name=product_name)
        required_text = req_text_tpl

        i2i_cases.append({
            "case_id": f"i2i_{idx:03d}",
            "track": "i2i",
            "instruction": instruction,
            "required_text": required_text,
            "target_layout": layout,
            "difficulty": difficulty,
            "text_length": text_len,
            "output_size": size,
            "reference_image": ref_path,
            "edit_type": edit_type
        })

    # Ghi file ra benchmarks/tendoo_v1/cases/
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    with open(CASES_DIR / "t2i.jsonl", "w", encoding="utf-8") as f:
        for c in t2i_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(CASES_DIR / "i2i.jsonl", "w", encoding="utf-8") as f:
        for c in i2i_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"🎉 Đã xây dựng {len(t2i_cases) + len(i2i_cases)} cases Benchmark v2 hoàn chỉnh ({len(t2i_cases)} T2I + {len(i2i_cases)} I2I)!")

if __name__ == "__main__":
    build_benchmark_v1_cases()
