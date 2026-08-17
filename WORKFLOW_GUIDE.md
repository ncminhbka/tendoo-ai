# HƯỚNG DẪN QUY TRÌNH LÀM VIỆC (AI ENGINEER WORKFLOW GUIDE)
**Dự án: Tendoo Media AI - Generative AI cho Banner/Poster Tiếng Việt**
*Chương trình Thực tập VDT 2026 - Viettel Telecom*

---

## 1. MÔ HÌNH HOẠT ĐỘNG (HYBRID WORKFLOW MODEL)

Dự án áp dụng mô hình **Local Development, Remote GPU Execution** để tối ưu hóa trải nghiệm lập trình và tận dụng tối đa hạ tầng phần cứng của Viettel:

```
┌──────────────────────────────────────┐          ┌──────────────────────────────────────┐
│  💻 LAPTOP CÁ NHÂN (LOCAL VS CODE)   │          │  ⚡ GPU SERVER (VIETTEL JUPYTERLAB) │
├──────────────────────────────────────┤          ├──────────────────────────────────────┤
│ - Viết & chỉnh sửa mã nguồn Python   │ Git Push │ - Đã cấp: 2x NVIDIA A30 (48GB VRAM)  │
│ - Chuẩn hóa dataset JSON / Prompts   │ ───────► │ - Môi trường Conda: tendoo_ai        │
│ - Kiểm tra cú pháp (Dry-run mode)   │ Git Pull │ - Nạp mô hình FLUX.2 klein / SD3.5   │
│ - Đẩy mã nguồn lên GitHub Repo       │ ◄─────── │ - Chạy Batch Inference & VietOCR     │
└──────────────────────────────────────┘          └──────────────────────────────────────┘
```

* **GitHub Repository**: `git@github.com:ncminhbka/tendoo-ai.git`

---

## 2. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT STRUCTURE)

```text
Tendoo Media AI/
├── .gitignore              # Bỏ qua file ảnh outputs và model weights nặng
├── WORKFLOW_GUIDE.md       # Tài liệu hướng dẫn quy trình làm việc (File này)
├── phat_bieu_bai_toan.txt  # Bài phát biểu đề tài, chỉ số đo lường (Metrics) & quy trình
├── findings.txt            # Nhật ký nghiên cứu các papers (TextDiffuser, EasyText, Glyph...)
├── prompt_test.txt         # Tập tập hợp prompts tiếng Việt đa ngành hàng (F&B, Fitness, Spa...)
└── src/                    # Thư mục mã nguồn chính (sẽ phát triển)
    ├── benchmark/          # Scripts tự động đo lường VietOCR, CER, Latency
    ├── models/             # Định nghĩa kiến trúc / Custom Layers PyTorch
    └── utils/              # Các hàm tiện ích xử lý ảnh, text mask, prompt parser
```

---

## 3. THAO TÁC HÀNG NGÀY (DAILY OPERATIONAL STEPS)

### Bước 1: Viết Code & Kiểm tra trên Laptop (VS Code)
1. Mở dự án trong VS Code trên laptop cá nhân.
2. Chỉnh sửa code, thêm features mới hoặc cập nhật prompt test trong `prompt_test.txt`.
3. Kiểm tra tính đúng đắn của code ở chế độ Dry-Run (chạy trên CPU với dữ liệu nhỏ).

### Bước 2: Đẩy Code lên GitHub Repo
Khi code trên laptop đã ổn định và không còn lỗi cú pháp, thực hiện push lên GitHub:
```bash
git add .
git commit -m "feat: cập nhật pipeline benchmark và prompt test mới"
git push origin main
```

### Bước 3: Cập nhật & Thực thi trên GPU Server (JupyterLab Nội bộ)
1. Truy cập vào giao diện JupyterLab trên trình duyệt máy công ty qua URL nội bộ do Mentor cấp.
2. **LƯU Ý LƯU TRỮ VĨNH VIỄN (PERSISTENT STORAGE)**:
   - Trong JupyterLab có 2 thư mục gắn với đĩa cứng cố định: **`persistent-data/`** hoặc **`work/`**.
   - **QUY TẮC BẮT BUỘC**: Mọi dự án Git (`tendoo-ai`), model weights tải về (`.safetensors`), dataset và kết quả benchmark **phải nằm bên trong `persistent-data/` hoặc `work/`**.
   - Dữ liệu nằm trong 2 thư mục này sẽ **KHÔNG BỊ MẤT** kể cả khi tắt máy, hết session hay container bị restart.
3. Mở một **Terminal** mới trong thư mục vĩnh viễn (`cd persistent-data/tendoo-ai` hoặc `cd work/tendoo-ai`).
4. Kích hoạt môi trường conda của dự án:
   ```bash
   conda activate tendoo_ai
   ```
5. Kéo mã nguồn mới nhất về Server:
   ```bash
   git pull origin main
   # (Hoặc tải file ZIP từ GitHub nếu mạng nội bộ chặn Git SSH)
   ```
6. Kiểm tra tình trạng GPU trước khi chạy:
   ```bash
   nvidia-smi
   ```

### Bước 4: Chạy Benchmark / Huấn luyện & Thu hoạch Kết quả
1. Chạy script thực thi công việc nặng (Inference / Evaluation / Training):
   ```bash
   python run_benchmark.py --model flux-klein --prompts prompt_test.txt
   ```
2. **Theo dõi GPU thời gian thực**: Mở một tab Terminal riêng và gõ `watch -n 1 nvidia-smi`.
3. **Xem kết quả**:
   - Mở thư mục `generated/` ngay trên giao diện JupyterLab để xem các bức ảnh Banner/Poster vừa được sinh ra.
   - Kiểm tra file log/báo cáo kết quả điểm số VietOCR CER (Character Error Rate) và thời gian xử lý (Latency).

---

## 4. QUẢN LÝ SESSION & QUY TRÌNH TẮT JUPYTERLAB AN TOÀN

### ❓ Trạng thái Session & Tiến trình:
- **Tệp tin (Files/Data)**: Lưu trong `persistent-data/` hoặc `work/` sẽ **giữ nguyên 100%**, không bao giờ mất.
- **Biến trong RAM / GPU VRAM**: Khi đóng trình duyệt hoặc tắt Kernel, các biến tạm trong RAM sẽ xóa (để trả GPU cho người khác).
- **Chạy job dài không lo mất session**: Nếu muốn chạy job huấn luyện dài (vài tiếng), dùng `tmux` hoặc `nohup`:
  ```bash
  tmux new -s train_job
  python train.py
  # Bấm Ctrl+B rồi bấm D để thoát ra ngoài mà job vẫn chạy ngầm
  ```

### 🛡️ Quy trình 4 Bước Tắt JupyterLab An toàn:

1. **Lưu toàn bộ công việc**:
   - Nhấn `Ctrl + S` hoặc menu `File -> Save All` trên tất cả các tab đang mở.
2. **Commit & Push kết quả mới (nếu có)**:
   - Trong Terminal: `git add . && git commit -m "results: cập nhật kết quả benchmark" && git push origin main`.
3. **Tắt các Kernel / Terminal giải phóng GPU (Cực kỳ quan trọng)**:
   - Nhấp vào biểu tượng **Running Terminals and Kernels** (hình ô vuông có dấu tròn 🟢 ở thanh công cụ dọc bên trái).
   - Nhấp **Shut Down** ở tất cả các Notebook/Terminal đang chạy ngầm để trả lại 100% VRAM GPU A30 cho hệ thống.
4. **Đăng xuất / Tắt Tab**:
   - Vào menu `File -> Log Out` (hoặc `File -> Shut Down`), sau đó đóng tab trình duyệt an toàn.

---

## 5. BEST PRACTICES & LƯU Ý KHI LÀM VIỆC

1. **Quản lý Tài nguyên VRAM**:
   - Khi load mô hình lớn (FLUX.2 klein-4B / SDXL), luôn dùng kiểu dữ liệu `torch.bfloat16` hoặc `torch.float16` để tiết kiệm VRAM.
   - Giải phóng bộ nhớ VRAM khi đổi mô hình bằng: `torch.cuda.empty_cache()`.
2. **Quản lý Môi trường Python**:
   - Tuyệt đối không cài đè thư viện vào môi trường `(base)`. Luôn làm việc trong `(tendoo_ai)`.
3. **Theo dõi Tiến độ với Mentor**:
   - Tổng hợp báo cáo đánh giá tự động (file CSV/Markdown điểm số Benchmark) gửi cho Mentor sau mỗi lần thử nghiệm phương án cải tiến mới.
