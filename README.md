# Tendoo Media AI - Viettel Telecom (VDT 2026)

## 📌 Giới Thiệu Dự Án
**Tendoo Media AI** là tính năng AI tạo và chỉnh sửa ảnh quảng cáo thương mại (Banner, Poster, Ảnh sản phẩm, Menu món, Tin tuyển dụng...) dành cho nền tảng quản lý bán hàng **Tendoo** do **Tổng Công ty Viễn thông Viettel (Viettel Telecom)** phát triển, phục vụ hộ kinh doanh cá thể, cửa hàng bán lẻ và doanh nghiệp vừa và nhỏ (SME).

## 🚀 Tính Năng Cốt Lõi
1. **Text-to-Image (T2I)**: Sinh mới poster quảng cáo, banner ưu đãi, menu món ăn từ câu lệnh Tiếng Việt tự nhiên với chữ hiển thị chính xác cao.
2. **Image-to-Image (I2I)**: Tải ảnh sản phẩm gốc, tự động thay phông nền studio, ghép bối cảnh lifestyle, bảo tồn logo thương hiệu và tạo banner Key Visual chuyên nghiệp.

## 📊 Bộ Benchmark TendooBizEval-Vi (`benchmarks/tendoo_v1/`)
Tập đề thi đánh giá năng lực mô hình bao gồm **111 cases** (65 T2I + 46 I2I):
- **Bao phủ 10 loại nhu cầu SME & Tiếng Việt chuyên sâu**:
  1. Poster Khuyến mại (Sale 50%, Giờ vàng...)
  2. Giới thiệu sản phẩm (Skincare, Mỹ phẩm, Điện tử...)
  3. Khai trương & Sự kiện (Grand Opening...)
  4. Feedback khách hàng (Đánh giá 5 sao...)
  5. Tuyển dụng nhân sự (Tuyển nhân viên, thu ngân...)
  6. Menu món ăn & Đồ uống (Phở bò gia truyền, Trà trái cây...)
  7. Sinh tự do & Văn hóa Việt (Tết 2026, Áo dài, Nón lá sen...)
  8. **STRESS TEST 1**: Dấu Tiếng Việt khó & phức tạp (`ẫ`, `ỡ`, `ẻ`, `ửng`, `nghễu`...)
  9. **STRESS TEST 2**: Số, Định dạng tiền tệ (`199.000đ`, `500.000Đ`) & Hotline (`0987.654.321`, `1900-8198`)
  10. **STRESS TEST 3**: Text đa dòng & đa cấp độ cỡ chữ (Tiêu đề lớn + Subtitle + Giá + Hotline)
- **Kích thước chuẩn FLUX.2**: Tuân thủ bội số 16 (`1024x1024`, `1024x1280`, `1280x720`, `1024x1536`, `1024x768`, `1088x1920`, `1200x624`).
- **OCR & Visual Quality Evaluation**: Tự động đo lường chữ Tiếng Việt qua VietOCR (CER/WER) và độ khớp prompt qua PickScore/CLIP score.

## 💻 Hướng Dẫn Sử Dụng Trên Server

```bash
# 1. Kéo mã nguồn mới nhất
git pull origin main

# 2. Kích hoạt môi trường
conda activate tendoo_ai

# 3. Kiểm tra dataset
python src/benchmark/validate_dataset.py

# 4. Chạy đánh giá Baseline
python src/benchmark/run_benchmark.py
```
