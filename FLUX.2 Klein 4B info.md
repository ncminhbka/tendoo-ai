<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

## Tổng quan

**FLUX.2 [klein] 4B** là mô hình rectified-flow transformer khoảng 4 tỷ tham số của Black Forest Labs, hỗ trợ đồng thời:

- Text-to-image.
- Image editing với một hoặc nhiều ảnh tham chiếu.
- Sinh ảnh và chỉnh sửa ảnh trong cùng một pipeline.
- Chạy local trên GPU tiêu dùng.
- License Apache 2.0, phù hợp cả nghiên cứu và sử dụng thương mại theo điều khoản license.[^1]

Điểm quan trọng nhất là phải phân biệt hai checkpoint:


| Checkpoint | Mục tiêu | Số bước mặc định | Guidance | Phù hợp |
| :-- | :-- | --: | --: | :-- |
| `FLUX.2-klein-4B` | Distilled, ưu tiên tốc độ | 4 | 1.0 | Production, preview, realtime |
| `FLUX.2-klein-base-4B` | Base/undistilled, linh hoạt hơn | 50 | 4.0 | Fine-tuning, LoRA, nghiên cứu |

Bảng trên tương ứng với ví dụ inference chính thức của hai model card.[^2][^1]

## Các tham số khi inference

### 1. `prompt`

Chuỗi mô tả nội dung cần sinh hoặc yêu cầu chỉnh sửa.

```python
prompt = """
A cinematic product photo of a red mechanical keyboard on a dark desk,
soft studio lighting, shallow depth of field
"""
```

Với image editing, prompt nên mô tả **thay đổi cần thực hiện**, không chỉ mô tả lại ảnh đầu vào:

```text
Replace the blue backpack with a black leather backpack.
Keep the person, pose, background, and lighting unchanged.
```

FLUX.2 có thể tạo chữ trong ảnh nhưng text dài hoặc yêu cầu chính xác vẫn có thể bị sai hoặc méo.[^1]

### 2. `height`, `width`

Kích thước ảnh đầu ra:

```python
height=1024,
width=1024,
```

Ví dụ chính thức dùng 1024 × 1024 cho cả bản distilled và base.[^2][^1]

Nên bắt đầu với:

- 1024 × 1024: chất lượng và độ ổn định tốt.
- 768 × 768: giảm VRAM và thời gian.
- Tỉ lệ khác 1:1: dùng khi tác vụ cần portrait hoặc landscape.

Với flow/diffusion model, nên ưu tiên kích thước tương thích với latent downsampling của VAE và tránh các kích thước quá nhỏ hoặc quá lớn khi chưa kiểm thử.

### 3. `guidance_scale`

Mức độ bám prompt:

```python
guidance_scale=1.0  # bản distilled
guidance_scale=4.0  # bản base
```

Thiết lập khuyến nghị từ model card:

- `FLUX.2-klein-4B`: `guidance_scale=1.0`.
- `FLUX.2-klein-base-4B`: `guidance_scale=4.0`.[^1][^2]

Không nên áp dụng giá trị của bản base cho bản distilled một cách máy móc. Bản distilled đã được huấn luyện để chạy với rất ít bước và guidance thấp; tăng guidance quá cao có thể làm ảnh kém tự nhiên hoặc giảm độ ổn định.

### 4. `num_inference_steps`

Số bước sampling:

```python
num_inference_steps=4   # distilled
num_inference_steps=50  # base
```

Đây là khác biệt lớn nhất giữa hai model:

- Bản distilled: dùng khoảng 4 bước, nhanh và phù hợp inference tương tác.
- Bản base: dùng khoảng 50 bước để tận dụng đầy đủ khả năng của checkpoint chưa distill.[^3]

Với bản distilled, tăng lên 20–50 bước thường không mang lại lợi ích tương ứng vì checkpoint đã được tối ưu cho quỹ đạo sampling ngắn. Với bản base, có thể thử khoảng 30–50 bước, nhưng 50 là cấu hình tham chiếu an toàn.

### 5. `generator`

Seed để tái lập kết quả:

```python
generator=torch.Generator(device="cuda").manual_seed(0)
```

- Cùng prompt, seed, kích thước và thông số thường cho kết quả có thể tái lập tương đối.
- Đổi seed để tạo các biến thể khác nhau.
- Khi benchmark, nên cố định seed.

Lưu ý: khác phiên bản PyTorch, CUDA, Diffusers hoặc thiết bị có thể làm kết quả thay đổi nhẹ.

### 6. `image` hoặc ảnh tham chiếu

Trong image editing, pipeline nhận thêm một hoặc nhiều ảnh tham chiếu. Ảnh được đưa vào cùng prompt để mô hình hiểu:

- Đối tượng cần giữ lại.
- Phong cách cần áp dụng.
- Vật thể cần thay thế.
- Thành phần cần kết hợp giữa nhiều ảnh.

FLUX.2 [klein] hỗ trợ text-to-image, chỉnh sửa một ảnh tham chiếu và chỉnh sửa nhiều ảnh tham chiếu.[^3]

Tên tham số cụ thể có thể khác tùy phiên bản Diffusers hoặc workflow ComfyUI. Vì vậy cần kiểm tra signature của pipeline đang cài:

```python
import inspect
from diffusers import Flux2KleinPipeline

print(inspect.signature(Flux2KleinPipeline.__call__))
```

Không nên tự giả định rằng mọi phiên bản đều dùng đúng tên `image`, `images`, `reference_images` hoặc `input_images`.

### 7. `torch_dtype`

Model card dùng BF16:

```python
dtype = torch.bfloat16
```

Đây là lựa chọn nên dùng trên GPU hỗ trợ BF16. Nếu GPU hoặc backend không hỗ trợ tốt BF16, có thể cân nhắc FP16/FP8 tùy checkpoint và framework, nhưng cần kiểm tra tính tương thích.

### 8. Offload và memory

Để giảm VRAM:

```python
pipe.enable_model_cpu_offload()
```

Model card ghi nhận bản 4B cần khoảng 13 GB VRAM trong cấu hình Diffusers đầy đủ; tài liệu ComfyUI báo các cấu hình FP8 có thể ở mức thấp hơn, khoảng 8.4 GB cho distilled và 9.2 GB cho base trên thiết lập được tài liệu hóa. Các con số này phụ thuộc độ phân giải, dtype, text encoder, VAE, quantization và cơ chế offload.[^4][^1]

## Inference bằng Diffusers

### Bản distilled 4B

Cài phiên bản Diffusers mới:

```bash
pip install git+https://github.com/huggingface/diffusers.git
```

Mã inference cơ bản:

```python
import torch
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=dtype,
)

pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device=device).manual_seed(0),
).images[^0]

image.save("flux-klein-4b.png")
```

Đây là cấu hình chính thức: BF16, 1024 × 1024, guidance 1.0 và 4 bước.[^1]

### Bản base 4B

```python
import torch
from diffusers import Flux2KleinPipeline

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-4B",
    torch_dtype=dtype,
)

pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=4.0,
    num_inference_steps=50,
    generator=torch.Generator(device=device).manual_seed(0),
).images[^0]

image.save("flux-klein-base-4b.png")
```

Bản base dùng guidance 4.0 và 50 bước theo ví dụ chính thức.[^2]

## Cách inference bằng repository chính thức

Repository chính thức cung cấp mã inference tối giản và CLI cho sinh ảnh cũng như chỉnh sửa ảnh. Quy trình tổng quát:

```bash
git clone https://github.com/black-forest-labs/flux2
cd flux2

python3.12 -m venv .venv
source .venv/bin/activate

pip install -e . \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --no-cache-dir
```

Có thể cấu hình đường dẫn model bằng biến môi trường:

```bash
export KLEIN_4B_MODEL_PATH="/path/to/FLUX.2-klein-4B"
export KLEIN_4B_BASE_MODEL_PATH="/path/to/FLUX.2-klein-base-4B"
export AE_MODEL_PATH="/path/to/flux2-vae"
```

Sau đó khởi chạy CLI:

```bash
PYTHONPATH=src python scripts/cli.py
```

Nếu không chỉ định đường dẫn, repository có thể tự tải weights theo cơ chế của nó.[^3]

## Cấu trúc model trong ComfyUI

Theo tài liệu ComfyUI, các thành phần 4B chính gồm:

```text
ComfyUI/
└── models/
    ├── text_encoders/
    │   └── qwen_3_4b.safetensors
    ├── diffusion_models/
    │   ├── flux-2-klein-4b-fp8.safetensors
    │   └── flux-2-klein-base-4b-fp8.safetensors
    └── vae/
        └── flux2-vae.safetensors
```

Trong đó:

- `qwen_3_4b.safetensors`: text encoder.
- `flux-2-klein-4b-fp8.safetensors`: distilled model.
- `flux-2-klein-base-4b-fp8.safetensors`: base model.
- `flux2-vae.safetensors`: VAE dùng chung cho workflow 4B.[^4]

ComfyUI có workflow riêng cho:

- Text-to-image distilled.
- Image edit distilled.
- Image edit base.
- Multi-reference editing.

Cần cập nhật ComfyUI đủ mới, vì các node FLUX.2 [klein] có thể chưa tồn tại trong bản stable cũ.[^4]

## Distilled hay Base?

### Chọn `FLUX.2-klein-4B`

Nên chọn bản distilled nếu bạn cần:

- Tốc độ cao.
- Preview tương tác.
- Sinh nhiều ảnh.
- Ứng dụng production hoặc API nội bộ.
- Chạy trên GPU tiêu dùng.
- Khoảng 4 bước inference.

Đây là lựa chọn mặc định cho hầu hết tác vụ sinh ảnh thông thường. Repository chính thức mô tả bản distilled là lựa chọn cho production và realtime.[^3]

### Chọn `FLUX.2-klein-base-4B`

Nên chọn bản base nếu bạn cần:

- Fine-tuning.
- LoRA training.
- Tùy biến sâu.
- Nhiều đa dạng đầu ra hơn.
- Kiểm soát sampling linh hoạt hơn.
- Nghiên cứu hoặc thử nghiệm pipeline.

Base không được step-distill hoặc guidance-distill, nên chạy chậm hơn nhưng linh hoạt hơn cho huấn luyện và nghiên cứu.[^2][^3]

## Cấu hình khuyến nghị

### Nhanh, ổn định

```python
height=1024
width=1024
guidance_scale=1.0
num_inference_steps=4
dtype=torch.bfloat16
```


### Chất lượng và đa dạng hơn

```python
height=1024
width=1024
guidance_scale=4.0
num_inference_steps=50
dtype=torch.bfloat16
```


### Khi thiếu VRAM

- Dùng FP8 hoặc quantized checkpoint phù hợp.
- Bật `enable_model_cpu_offload()`.
- Giảm độ phân giải xuống 768 hoặc 896.
- Giảm batch size về 1.
- Không chạy đồng thời nhiều pipeline.
- Tách text encoder hoặc model sang CPU nếu framework hỗ trợ.

Các con số VRAM không cố định: model card ghi khoảng 13 GB cho cấu hình 4B, trong khi workflow FP8 của ComfyUI có mức yêu cầu thấp hơn.[^4][^1]

## Những điểm dễ nhầm

- `4B` không đồng nghĩa với mọi checkpoint 4B có cách sampling giống nhau.
- `FLUX.2-klein-4B` distilled dùng `4 steps / guidance 1.0`.
- `FLUX.2-klein-base-4B` dùng khoảng `50 steps / guidance 4.0`.
- Không nên dùng bản base cho mục tiêu realtime nếu không cần sự linh hoạt của base.
- Không nên đánh giá chất lượng bằng cách dùng 4 bước cho bản base.
- Không nên đánh giá tốc độ distilled bằng cách tăng tùy tiện lên 50 bước.
- Tên tham số image editing phụ thuộc phiên bản Diffusers; cần kiểm tra API thực tế.
- Khi deploy ứng dụng, nên bổ sung bộ lọc nội dung, metadata provenance/C2PA hoặc watermark theo yêu cầu sản phẩm. Repository và model card đều đề cập đến watermark và cơ chế provenance.[^1][^3]
<span style="display:none">[^10][^11][^12][^13][^14][^15][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B

[^2]: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B

[^3]: https://github.com/black-forest-labs/flux2

[^4]: https://docs.comfy.org/tutorials/flux/flux-2-klein

[^5]: https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8

[^6]: https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF

[^7]: https://huggingface.co/tonera/FLUX.2-klein-4B-fp8-diffusers

[^8]: https://huggingface.co/Runpod/FLUX.2-klein-4B-mflux-4bit

[^9]: https://huggingface.co/YuCollection/FLUX.2-klein-4B-bf16

[^10]: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/main

[^11]: https://bfl.ai/models/flux-2-klein

[^12]: https://huggingface.co/unsloth/FLUX.2-klein-base-4B-GGUF

[^13]: https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_training/full/FLUX.2-klein-base-4B.sh

[^14]: https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/flux2/model_inference/FLUX.2-klein-4B.py

[^15]: https://github.com/VladimirRL/flux2-klein-4b/tree/main

