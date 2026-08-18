# TendooBizEval-Vi v1 Benchmark Specification

## Overview
`tendoo_v1` là bộ Benchmark Suite hai nhánh (Text-to-Image và Image-to-Image) được thiết kế riêng để đánh giá năng lực của các mô hình sinh ảnh AI (như FLUX.2, SDXL, PixArt) trên đồ họa thương mại Tiếng Việt, Banner quảng cáo và Ảnh sản phẩm dành cho hộ kinh doanh & SME Tendoo (Viettel Telecom).

## Tracks & Use Cases
1. **T2I (Text-to-Image)**: 50 câu lệnh đề thi Tiếng Việt tự nhiên bao phủ 7 nhóm nhu cầu thực tế của SME Tendoo:
   - **Poster Khuyến Mại & Ưu Đãi**: Sale 50%, Giờ Vàng Giá Sốc, Mua 1 Tặng 1, Voucher 50K...
   - **Ảnh Giới Thiệu Sản Phẩm**: Kem chống nắng, Skincare, Nước hoa, Tai nghe, Giày sneaker...
   - **Banner Khai Trương & Sự Kiện**: Grand Opening, Mừng khai trương chi nhánh...
   - **Ảnh Feedback Khách Hàng**: Góc Feedback khách yêu, Đánh giá 5 sao...
   - **Ảnh Tin Tuyển Dụng**: Tuyển nhân viên bán hàng, pha chế, thu ngân...
   - **Menu Món Ăn & Đồ Uống**: Thực đơn Phở Bò gia truyền, Menu trà trái cây...
   - **Sinh Tự Do & Văn Hóa Việt**: Tết 2026 hoa đào, Áo dài nón lá sen, Phố cổ Hà Nội, Gốm Bát Tràng...

2. **I2I (Image-to-Image / Product Placement)**: 46 câu lệnh đề thi dựa trên 46 ảnh reference sản phẩm thương mại thật, phân bổ đều qua 5 nhiệm vụ chỉnh sửa ảnh (Edit Types):
   - `background_replacement`: Thay nền studio sang trọng, bãi biển...
   - `lifestyle_placement`: Ghép sản phẩm vào bàn gỗ, gian bếp, góc làm việc...
   - `key_visual`: Thiết kế banner Key Visual hiệu ứng neon nổi bật...
   - `preserve_packaging_logo`: Giữ nguyên logo & kiểu dáng bao bì gốc...
   - `object_removal_cleanup`: Tách sản phẩm khỏi vật thể xao nhãng, đặt lên bục gỗ...

## FLUX.2 Technical Specifications Compliance
Tất cả kích thước ảnh đầu ra tuân thủ nghiêm ngặt quy định bội số của 16 của mô hình FLUX.2 Klein:
- `1024x1024` (1:1 Square)
- `1024x1280` (4:5 Post)
- `1280x720` (16:9 Banner)
- `1024x1536` (2:3 Poster)
- `1024x768` (4:3 Standard)
- `1088x1920` (9:16 Mobile Story)
- `1200x624` (1.91:1 Landscape)

## Structure
- `schema/`: Quy định JSON Schema chuẩn (Draft-07) cho đề thi (`case.schema.json`) và kết quả (`result.schema.json`).
- `cases/`: File dataset `t2i.jsonl` và `i2i.jsonl`.
- `references/`: Thư mục lưu trữ 46 ảnh sản phẩm HD theo 5 ngành hàng (`beauty`, `food_beverage`, `fashion`, `home_electronics`, `stationery_office`).
- `manifests/`: Registry `reference_manifest.json` đăng ký danh mục ảnh sản phẩm.
- `outputs/`: Nơi lưu ảnh sinh ra từ 3 Seed ngẫu nhiên (42, 43, 44) và log `result.jsonl`.
- `reports/`: Xuất báo cáo tổng hợp `report.csv` và `report.md`.

## Execution Commands

### 1. Kiểm Tra Tính Toàn Vẹn Dataset:
```bash
python src/benchmark/validate_dataset.py
```

### 2. Chạy Đánh Giá Baseline Chính Thức:
```bash
python src/benchmark/run_benchmark.py
```
