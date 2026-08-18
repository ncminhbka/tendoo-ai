# BÁO CÁO KẾT QUẢ BENCHMARK TENDOOBIZEVAL-VI (v0)

**Thời gian xuất báo cáo**: Tue 08/18/2026
**Tổng số lượt chạy hiện có (3 Seeds/case)**: `15`
**Dataset hiện tại**: `50 T2I + 50 I2I`
**Trạng thái dữ liệu**: `CẦN CHẠY LẠI TOÀN BỘ SAU KHI ĐỔI CASES`

---

## 📊 BẢNG TỔNG HỢP HIỆU NĂNG & CHẤT LƯỢNG

| Chỉ số Đánh giá | Giá trị Thực tế | Mục tiêu Đặt ra | Trạng thái |
| :--- | :---: | :---: | :---: |
| **Tỷ lệ Đạt (Pass Rate)** | `0.0%` | `>= 70.0%` | 🟡 CẦN CẢI TIẾN |
| **Điểm T2I Composite Score** | `30.0 / 100` | `>= 75.0` | 🟡 CẦN CẢI TIẾN |
| **Điểm I2I Composite Score** | `0.0 / 100` | `>= 75.0` | 🟡 CẦN CẢI TIẾN |
| **Độ trễ trung bình (Mean Latency)** | `0.63s / ảnh` | `< 15.0s` | 🟢 RẤT TỐT |
| **Lượt lỗi Text (CER > 30%)** | `15` | `0` | 🔴 LỖI TEXT |
| **Case chờ ảnh Ref (Pending)** | `0` | `0` | ℹ️ TRẠNG THÁI |

---

## ⚡ PHÂN TÍCH CHI TIẾT ĐỘ TRỄ (LATENCY) & THÔNG LƯỢNG (THROUGHPUT)

| Chỉ số Hiệu năng Runtime | Giá trị Thực tế | Ý nghĩa Kỹ thuật |
| :--- | :---: | :--- |
| **Mean Latency (Độ trễ TB)** | **`0.63 giây/ảnh`** | Thời gian trung bình để sinh 1 ảnh |
| **P95 Latency (Phân vị 95%)** | **`2.5 giây/ảnh`** | 95% số ảnh sinh ra nhanh hơn ngưỡng này |
| **Min / Max Latency** | **`0.5s / 2.5s`** | Khoảng thời gian sinh ảnh nhanh nhất / chậm nhất |
| **Throughput (Thông lượng)** | **`95.2 ảnh/phút`** | Số lượng ảnh sinh ra trong 1 phút trên GPU A30 |
| **Peak VRAM Allocation** | **`0.0 GB`** | Dung lượng bộ nhớ VRAM đỉnh điểm chiếm dụng |

---

## 📋 ĐIỂM CHI TIẾT TỪNG TRACK

### 1. Track T2I (Text-to-Image Poster Ad)
- **Dataset**: `50 cases`; **runs hiện có**: `5 cases`
- **Trọng số**: Text Accuracy 40%, Alignment 25%, Aesthetic 20%, Layout 15%
- **Điểm trung bình**: **`30.0 / 100`**

### 2. Track I2I (Product Placement & Image Editing)
- **Dataset**: `50 cases`; **runs hiện có**: `0 cases`
- **Trọng số**: Product Preservation 30%, Text Accuracy 25%, Aesthetic 20%, BG Integration 15%, Instruction 10%
- **Điểm trung bình**: **`0.0 / 100`**

## 📐 PHÂN TÍCH THEO CASE METADATA

| Track | Dimension | Value | Runs | Avg Score | Avg Latency |
| :--- | :--- | :--- | ---: | ---: | ---: |
| t2i | category | unknown | 15 | 30.00 | 0.63s |
| t2i | difficulty | easy | 6 | 30.00 | 0.83s |
| t2i | difficulty | hard | 3 | 30.00 | 0.50s |
| t2i | difficulty | medium | 6 | 30.00 | 0.50s |
| t2i | layout | horizontal_16x9 | 3 | 30.00 | 0.50s |
| t2i | layout | horizontal_4x3 | 3 | 30.00 | 0.50s |
| t2i | layout | square_1x1 | 3 | 30.00 | 1.17s |
| t2i | layout | vertical_2x3 | 3 | 30.00 | 0.50s |
| t2i | layout | vertical_4x5 | 3 | 30.00 | 0.50s |
| t2i | output_size | 1024x1024 | 3 | 30.00 | 1.17s |
| t2i | output_size | 1024x1280 | 3 | 30.00 | 0.50s |
| t2i | output_size | 1024x1536 | 3 | 30.00 | 0.50s |
| t2i | output_size | 1024x768 | 3 | 30.00 | 0.50s |
| t2i | output_size | 1280x720 | 3 | 30.00 | 0.50s |
| t2i | text_length | long | 3 | 30.00 | 0.50s |
| t2i | text_length | medium | 6 | 30.00 | 0.50s |
| t2i | text_length | short | 6 | 30.00 | 0.83s |

---

*Báo cáo được khởi tạo tự động bởi hệ thống TendooBizEval-Vi Framework.*
