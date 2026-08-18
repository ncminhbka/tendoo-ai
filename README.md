# Tendoo Media AI - Viettel Telecom (VDT 2026)

## 📌 Giới Thiệu Dự Án
**Tendoo Media AI** là tính năng AI tạo và chỉnh sửa ảnh quảng cáo thương mại (Banner, Poster, Ảnh sản phẩm, Menu món, Tin tuyển dụng...) dành cho nền tảng quản lý bán hàng **Tendoo** do **Tổng Công ty Viễn thông Viettel (Viettel Telecom)** phát triển, phục vụ hộ kinh doanh cá thể, cửa hàng bán lẻ và doanh nghiệp vừa và nhỏ (SME).

## 🚀 Tính Năng Cốt Lõi
1. **Text-to-Image (T2I)**: Sinh mới poster quảng cáo, banner ưu đãi, menu món ăn từ câu lệnh Tiếng Việt tự nhiên với chữ hiển thị chính xác cao.
2. **Image-to-Image (I2I)**: Tải ảnh sản phẩm gốc, tự động thay phông nền studio, ghép bối cảnh lifestyle, bảo tồn logo thương hiệu và tạo banner Key Visual chuyên nghiệp.

## 📊 Bộ Benchmark TendooBizEval-Vi (`benchmarks/tendoo_v1/`)
Tập đề thi đánh giá năng lực mô hình bao gồm **96 cases** (50 T2I + 46 I2I):
- **Bao phủ 7 loại nhu cầu SME**: Khuyến mại sale 50%, Giới thiệu sản phẩm, Khai trương chi nhánh, Feedback khách hàng, Tuyển dụng nhân sự, Menu món ăn, Văn hóa Việt (Tết 2026, Áo dài, Nón lá sen...).
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
