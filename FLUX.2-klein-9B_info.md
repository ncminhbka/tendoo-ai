# Thông tin chi tiết Mô hình FLUX.2-klein-9B (Black Forest Labs)

## 1. Tổng quan về Mô hình
* **Tên mô hình**: `FLUX.2 [klein] 9B`
* **ID trên Hugging Face / API**: `black-forest-labs/FLUX.2-klein-9B`
* **Đơn vị phát triển**: **Black Forest Labs (BFL)** – công ty sáng lập bởi các tác giả gốc của Stable Diffusion (CompVis / Runway / Stability AI).
* **Ý nghĩa tên gọi**: "Klein" trong tiếng Đức có nghĩa là "nhỏ" (small), đại diện cho dòng mô hình sinh ảnh gọn nhẹ, tối ưu cho tốc độ phản hồi cực nhanh (sub-second / real-time).

---

## 2. Kiến trúc & Thông số Kỹ thuật
* **Số lượng tham số (Parameters)**: **9 tỷ tham số (9B)**.
* **Kiến trúc cốt lối**: **Rectified Flow Transformer** kết hợp với Text Encoder **Qwen3 8B**, giúp khả năng hiểu câu lệnh (prompt adherence) đạt độ chính xác cao.
* **Kỹ thuật Chưng cất (Distillation)**: Mô hình được distilled đặc biệt giúp sinh ảnh chất lượng photorealistic sắc nét chỉ sau vài bước lấy mẫu (inference steps).
* **Tốc độ xử lý**: Sinh ảnh hoặc chỉnh sửa ảnh chỉ trong khoảng **0.5 - 1.0 giây** trên phần cứng GPU hiện đại hoặc qua API server tối ưu.

---

## 3. Khả năng Chỉnh sửa & Số lượng Ảnh Tham chiếu (Image Reference Capabilities)

* **Số lượng ảnh tham chiếu hỗ trợ**: Dòng mô hình **FLUX.2 hỗ trợ tối đa 10 ảnh tham chiếu (Up to 10 Reference Images)** trong cùng 1 yêu cầu (Single & Multi-Reference Editing).
* **Các trường hợp ứng dụng (Use-cases)**:
  1. **Image-to-Image / Photo Editing**: Truyền 1 ảnh gốc và thêm/bớt/thay đổi chi tiết (ví dụ: thêm kính râm, đổi kiểu tóc, đổi màu áo).
  2. **Consistent Character (Giữ nét nhân vật)**: Truyền ảnh nhân vật mẫu để sinh ra các ảnh mới có cùng khuôn mặt/phong cách.
  3. **Multi-Reference Style Transfer**: Truyền 1 ảnh chứa nhân vật + 1 ảnh chứa phong cách nghệ thuật/bối cảnh + 1 ảnh trang phục để kết hợp thành ảnh hoàn chỉnh.

---

## 4. Thông số Cấu hình Khuyến nghị (Recommended Parameters)

| Tham số | Giá trị khuyến nghị | Ghi chú |
| :--- | :--- | :--- |
| **`num_inference_steps`** | **4 – 5** | Do mô hình đã được chưng cất, 4-5 bước là tối ưu nhất cả về tốc độ và chất lượng. |
| **`guidance_scale`** | **1.0 – 4.0** | Giá trị mặc định là 4.0 (hoặc 1.0 khi sử dụng qua vLLM gateway). |
| **`seed`** | Số nguyên ngẫu nhiên (ví dụ: 42) | Cố định seed giúp tái tạo lại kết quả sinh ảnh chính xác. |
| **Độ phân giải (Resolution)** | `1024x1024`, `768x1024`, `1024x768`, `576x1024`, `1024x576` | Tương thích tốt nhất với tỉ lệ 1:1, 3:4, 4:3, 9:16, 16:9. |

---

## 5. Giấy phép Sử dụng (Licensing)
* **Loại giấy phép**: **FLUX Non-Commercial License** (Giấy phép phi thương mại của Black Forest Labs).
* **Quyền hạn**: Được phép sử dụng cho mục đích nghiên cứu, học tập, thử nghiệm cá nhân và phát triển ứng dụng phi thương mại. *(Lưu ý: Bản FLUX.2 [klein] 4B nhỏ hơn được phát hành theo giấy phép Apache 2.0 mở hoàn toàn).*

---

## 6. Hướng dẫn Gọi API & Mã Nguồn Mẫu (Python Client)

### API Endpoint Format (OpenAI Chat Completions Compatible)
* **URL**: `https://ai-api.tendoo.vn/llm-gw/vllm/chat/completions`
* **Method**: `POST`
* **Headers**:
  * `Authorization`: `Bearer <YOUR_TOKEN>`
  * `Content-Type`: `application/json`
  * `x-api-key`: `<X_API_KEY>`
  * `apikey`: `<API_KEY>`

### Python Code Snippet (Hỗ trợ cả Sinh ảnh mới & Chỉnh sửa ảnh tham chiếu)

```python
from flux_client import FluxClient

client = FluxClient()

# 1. Sinh ảnh mới từ prompt
client.generate_image(
    prompt="Sinh ảnh một cô gái sinh đẹp trong công viên", 
    output_path="girl.png"
)

# 2. Chỉnh sửa ảnh có ảnh tham chiếu (Image Reference)
client.generate_image(
    prompt="Thêm một chiếc kính râm thời trang màu đen lên mặt cô gái trong ảnh", 
    output_path="edited_girl.png",
    reference_images=["girl.png"] # Hỗ trợ danh sách tối đa 10 ảnh
)
```
