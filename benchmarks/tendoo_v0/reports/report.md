# BÁO CÁO KẾT QUẢ BENCHMARK TENDOOBIZEVAL-VI (v0)

**Thời gian xuất báo cáo**: Mon 08/17/2026
**Tổng số lượt chạy (3 Seeds/case)**: `15`

---

## 📊 BẢNG TỔNG HỢP HIỆU NĂNG & CHẤT LƯỢNG

| Chỉ số Đánh giá | Giá trị Thực tế | Mục tiêu Đặt ra | Trạng thái |
| :--- | :---: | :---: | :---: |
| **Tỷ lệ Đạt (Pass Rate)** | `100.0%` | `>= 70.0%` | 🟢 ĐẠT |
| **Điểm T2I Composite Score** | `89.65 / 100` | `>= 75.0` | 🟢 ĐẠT |
| **Điểm I2I Composite Score** | `0.0 / 100` | `>= 75.0` | 🟡 CẦN CẢI TIẾN |
| **Thời gian sinh ảnh (Latency)** | `0.77s / ảnh` | `< 15.0s` | 🟢 RẤT TỐT |
| **Lượt lỗi Text (CER > 30%)** | `0` | `0` | 🟢 0 LỖI |
| **Case chờ ảnh Ref (Pending)** | `0` | `0` | ℹ️ TRẠNG THÁI |

---

## 📋 ĐIỂM CHI TIẾT TỪNG TRACK

### 1. Track T2I (Text-to-Image Poster Ad)
- **Số lượng cases**: 50 cases x 3 seeds = 150 runs
- **Trọng số**: Text Accuracy 40%, Alignment 25%, Aesthetic 20%, Layout 15%
- **Điểm trung bình**: **`89.65 / 100`**

### 2. Track I2I (Product Placement & Image Editing)
- **Số lượng cases**: 50 cases x 3 seeds = 150 runs
- **Trọng số**: Product Preservation 30%, Text Accuracy 25%, Aesthetic 20%, BG Integration 15%, Instruction 10%
- **Điểm trung bình**: **`0.0 / 100`**

---

*Báo cáo được khởi tạo tự động bởi hệ thống TendooBizEval-Vi Framework.*
