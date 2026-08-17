# BÁO CÁO LỰA CHỌN TOP 3 FOUNDATION MODELS CHO TENDOO MEDIA AI

## 🎯 Tiêu chí Lựa chọn
Hệ thống Foundation Model được lựa chọn dựa trên 8 tiêu chí kỹ thuật khắt khe phục vụ tự host (Local Hosting) trên **2x NVIDIA A30 (48GB VRAM)** và nghiên cứu cải tiến kiến trúc:
1. **Commercial-friendly License** (Apache 2.0 / OpenRAIL-M - dùng bán hàng thương mại tại Viettel)
2. **Open Weights** (Sẵn có trên HuggingFace)
3. **Latency Hợp lý** (Sinh ảnh < 10-15s trên GPU A30)
4. **Architecture Code Open** (Code PyTorch kiến trúc mở 100%)
5. **Training Code Open** (Có sẵn code pre-training từ đầu và SFT/LoRA chính thức)
6. **Fine-tuning Support** (Hỗ trợ LoRA / QLoRA / Full SFT)
7. **Modification Feasible** (Dễ dàng sửa code PyTorch lõi: chèn Character Tokenizer, Position Embeddings, đổi Text Encoder sang ViT5/PhoBERT)
8. **Strong Text/Image Capability** (Khả năng vẽ chữ và giữ dáng sản phẩm)

---

## 🏆 TOP 3 FOUNDATION MODELS ĐƯỢC CHỌN

### 🥇 TOP 1: PixArt-Sigma (Huawei / OpenDataLab)
- **Kiến trúc**: Pure Diffusion Transformer (DiT) 0.6B - 2B params + T5 Text Encoder.
- **Giấy phép**: **Apache 2.0** (Thương mại 100%).
- **Lý do chọn**: 
  - **Mã nguồn mở 100%** cả weights lẫn code pre-training / SFT / LoRA trên GitHub (`PixArt-alpha/PixArt-sigma`).
  - Code PyTorch cực kỳ tối giản và tường minh, **dễ can thiệp sửa đổi kiến trúc nhất thế giới hiện nay**. 
  - Rất phù hợp để thực nghiệm chèn module **Character Position Encoding** (như paper *EasyText*) hoặc thay Text Encoder sang **ViT5 / PhoBERT**.
  - Dung lượng nhẹ, chạy siêu nhanh (< 2s/ảnh), tốn ít VRAM trên GPU A30.

### 🥈 TOP 2: HunyuanDiT (Tencent)
- **Kiến trúc**: DiT + Multilingual T5 (mT5-XXL) + Bilingual CLIP.
- **Giấy phép**: **Apache 2.0** (Thương mại 100%).
- **Lý do chọn**:
  - Code nghiên cứu mở 100% từ Tencent trên GitHub (`Tencent/HunyuanDiT`).
  - Tích hợp sẵn **mT5 (Multilingual T5)** nên khả năng phân tách ký tự và biểu diễn tiếng Việt có dấu (`ươ, ă, đ, ệ, ữ`) tốt hơn hẳn các model dùng T5 tiếng Anh thuần.
  - Hỗ trợ đầy đủ công cụ Fine-tuning SFT & LoRA.

### 🥉 TOP 3: SDXL 1.0 (Stable Diffusion XL Base 1.0)
- **Kiến trúc**: U-Net 2.6B + Dual CLIP (OpenCLIP ViT-bigG + CLIP ViT-L).
- **Giấy phép**: **OpenRAIL-M** (Cho phép dùng thương mại).
- **Lý do chọn**:
  - Mã nguồn U-Net mở 100%, hệ sinh thái công cụ Fine-tuning (ControlNet, LoRA, DreamBooth) phong phú nhất lịch sử AI.
  - Hầu hết các paper nghiên cứu text rendering (*Glyph-ByT5*, *EasyText*, *TextDiffuser*) đều có sẵn code can thiệp trên SDXL.
  - Làm **mô hình baseline đối chứng** lý tưởng giữa dòng kiến trúc U-Net và dòng DiT.

---

## 📌 Ghi chú về FLUX.1 [schnell] / FLUX.2 [klein] 4B
- **Vai trò**: Được giữ lại làm **Baseline Tham chiếu Thương mại (Commercial Baseline)** để so sánh điểm số VietOCR CER và Aesthetic Quality.
- **Hạn chế**: Mặc dù vẽ chữ rất đẹp và có giấy phép Apache 2.0, nhưng code pre-training nội bộ bị đóng kín và mô hình bị Distill nên **khó can thiệp sửa đổi kiến trúc lõi**.
