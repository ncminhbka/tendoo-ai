# FLUX.2 — Ghi chú kiến trúc đã xác nhận (dùng làm căn cứ nâng cấp TextGuider)

Tài liệu này liệt kê các sự thật đã **xác nhận từ nguồn chính thức** (repo
`black-forest-labs/flux2`, tài liệu Diffusers `Flux2Transformer2DModel`,
`model_index.json` của `FLUX.2-klein-base-4B-Diffusers`), đối chiếu với
những gì code TextGuider cũ đã **giả định sai hoặc chưa verify**. Mọi thay
đổi trong lần nâng cấp này đều truy được về một dòng trong bảng dưới.

## 1. Danh tính model / pipeline (xác nhận qua `model_index.json`)

| Thành phần | Giá trị thật | Code cũ giả định |
|---|---|---|
| Pipeline | `Flux2KleinPipeline` | đúng |
| Transformer | `Flux2Transformer2DModel` (diffusers) | đúng (đã dùng đúng tên) |
| Text encoder | `Qwen3ForCausalLM` | đúng model, nhưng **cách dùng sai** (xem mục 3) |
| Tokenizer | `Qwen2Tokenizer` (runtime checkpoint; có thể biến thể Fast tùy package) | encode_prompt thực tế dùng đường tokenize raw, padding tới 512 |
| VAE | `AutoencoderKLFlux2` | không tương tác trực tiếp |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | đúng |

Nguồn: `huggingface.co/YuCollection/FLUX.2-klein-base-4B-Diffusers/blob/main/model_index.json`

## 2. Cấu hình transformer thật (xác nhận qua doc `Flux2Transformer2DModel`)

```
Flux2Transformer2DModel(
    patch_size=1, in_channels=128, out_channels=None,
    num_layers=8,            # double-stream blocks — ĐỌC ĐỘNG, đừng hardcode
    num_single_layers=48,    # single-stream blocks — ĐỌC ĐỘNG, đừng hardcode
    attention_head_dim=128, num_attention_heads=48,   # số này là của [dev]; Klein 4B nhỏ hơn
    joint_attention_dim=15360,   # = 3 × d_model của [dev]; Klein 4B ≈ 7680 (3×2560)
    guidance_embeds=True,
)
```

**Hệ quả sửa code:**
- `textguider.py`/`textguider_attention.py` từng hardcode `num_heads=24`,
  `"5 dual-stream / 20 single-stream blocks"` — đều là số **đoán, không có
  nguồn**. Bản mới đọc `transformer.config.num_layers`,
  `transformer.config.num_single_layers`, `transformer.config.num_attention_heads`
  trực tiếp từ checkpoint đã load, không hardcode gì cả.
- `forward()` của `Flux2Transformer2DModel` **không có tham số
  `pooled_projections`** (khác FLUX.1/SD3). Xác nhận thêm bởi tài liệu
  huấn luyện bên thứ ba: *"No pooled_projections: FLUX.2 uses
  Flux2TimestepGuidanceEmbeddings which only takes timestep + guidance
  (no pooled text features)."* → code cũ ở `_full_textguider_generate`
  unpack `prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(...)`
  (3 giá trị) là **sai chữ ký thật**; `encode_prompt` của Flux2 trả về
  2 giá trị `(prompt_embeds, txt_ids)`. Đã sửa toàn bộ pipeline để dùng
  đúng 2-tuple và bỏ hẳn mọi xử lý `pooled_projections`.
- `guidance_embeds=True` trong config **không có nghĩa mọi forward đều phải
  nhận tensor `guidance`**. Implementation chính thức của
  `Flux2KleinPipeline` truyền `guidance=None` ở cả conditional và
  unconditional CFG forward; tensor này chỉ dành cho đường sampling native
  non-CFG khi gọi `denoise()`. CFG scale vẫn được áp dụng sau khi có hai
  `noise_pred`.

## 3. Text encoder Klein — Qwen3 dùng như bộ trích đặc trưng, KHÔNG phải chat model

Theo tài liệu bên thứ ba mô tả `Qwen3Embedder` trong `text_encoder.py`
(repo chính thức): đây là lớp **encoding-only**, lấy hidden state ở 3 tầng
`[9, 18, 27]` rồi nối lại (`b l (c d)`) thành embedding `3×d_model` mỗi
token — **không phải một lượt sinh chat với `apply_chat_template`**.

**Hệ quả sửa code:** `TextGuiderTokenParser.parse_tokens` cũ tự áp
`tokenizer.apply_chat_template(..., role="user", enable_thinking=False)`
rồi tokenize với `max_length=512, padding="max_length"` — đây là một lượt
tokenize **hoàn toàn tách biệt** với lượt mà `Qwen3Embedder`/`encode_prompt`
thật sự dùng. Nếu hai lượt không khớp tuyệt đối (template, độ dài, padding),
`quo_indices`/`text_token_indices` sẽ trỏ sai vị trí — lỗi im lặng nghiêm
trọng nhất trong toàn bộ codebase cũ.

Bản mới:
1. Mặc định tokenize **thô, không chat template** (khớp với việc Qwen3 chỉ
   được dùng làm feature extractor).
2. Sau khi có `encoder_hidden_states` thật, **bắt buộc assert** độ dài
   token khớp (`encoder_hidden_states.shape[1] == len(offsets sau lọc
   attention_mask)`), nếu lệch thì **raise lỗi rõ ràng** thay vì âm thầm
   chạy guidance sai.
3. Vẫn giữ đường vòng "thử chat-template nếu raw không khớp" cho các biến
   thể model có thể áp template, nhưng luôn qua bước assert ở (2).

## 4. Base model dùng CFG thật, không phải guidance-embedding một lượt

Repo chính thức có **hai hàm sampling khác nhau** (`src/flux2/sampling.py`):
- `denoise()` — dùng cho **model đã distill** (guidance=1.0, một lượt forward)
- `denoise_cfg()` — dùng cho **model Base** (guidance mặc định 4.0,
  **hai lượt forward: có-điều-kiện và không-điều-kiện**, kết hợp theo
  công thức CFG kinh điển)

`FLUX.2 [klein] 4B Base` (model mục tiêu của bạn) là **Base, không distill**
→ phải dùng đường `denoise_cfg`, tức là **CFG thật** (âm bản/uncond prompt
+ hai lượt transformer + trộn theo `guidance_scale`), **không phải** một
lượt forward với "guidance" nhúng sẵn.

**Hệ quả sửa code:** toàn bộ code cũ (`_full_textguider_generate` và
callback path) chỉ gọi transformer **một lượt duy nhất** mỗi bước, không
có prompt âm bản, không có công thức trộn CFG nào. Với model Base, đây là
thiếu sót ảnh hưởng trực tiếp tới chất lượng ảnh (độc lập với TextGuider).
Bản mới thêm encode prompt rỗng (uncond) + hai lượt forward + trộn CFG
đúng công thức `denoise_cfg`.

## 5. Patch/pack latent — số liệu khớp, giữ nguyên nhưng ưu tiên hàm gốc

`SimpleTuner` (tài liệu fine-tune FLUX.2 bên thứ ba) xác nhận: *"Latent
Channels: 32 VAE channels → 128 after pixel shuffle"* — khớp chính xác với
giả định cũ trong `latent_utils.py` (`D // 4 = 32`). Phần này **không sai**,
nhưng vẫn tiềm ẩn rủi ro nếu quy ước permute không khớp bit-for-bit với
hàm nội bộ của pipeline. Bản mới: **ưu tiên gọi `pipe._pack_latents` /
`pipe._unpack_latents`** nếu pipeline có sẵn (đảm bảo khớp tuyệt đối);
`latent_utils.py` chỉ còn là fallback khi không có pipeline (chế độ
`compute_attention_maps_native` với model gốc BFL).

## 6. Double-stream vs single-stream block — nhận diện theo cấu trúc, không đoán tên class

Không tìm được xác nhận 100% tên class `Attention`/`Processor` cụ thể của
Diffusers cho Flux2 (`Flux2Attention` như code cũ đoán **chưa được xác
nhận**). Tuy nhiên tài liệu nguồn Diffusers có nhắc tới `attn.to_added_qkv`
là phép chiếu QKV riêng cho token văn bản — **chỉ tồn tại ở block
double-stream** (single-stream dùng QKV+MLP gộp chung một phép chiếu).

**Hệ quả sửa code:** thay vì lọc theo chuỗi tên module
(`"transformer_blocks." in name and "single_" not in name` — đoán theo
quy ước của FLUX.1, chưa chắc đúng với Flux2), bản mới lọc theo **thuộc
tính cấu trúc**: `hasattr(module, "to_added_qkv")` → đây là double-stream
block. Cách này bền hơn trước thay đổi tên nội bộ giữa các phiên bản
Diffusers.

## 7. Timestep scaling — thống nhất một nguồn duy nhất

Code cũ có **3 chỗ** truyền timestep vào transformer với 2 kiểu scale
khác nhau trong cùng một bước denoise (`t` thô vs `t / 1000`) — tự mâu
thuẫn ngay trong `_full_textguider_generate`. Bản mới có một hàm
`_prepare_timestep()` duy nhất, dùng lại ở mọi lệnh gọi transformer trong
cùng một bước, đảm bảo lượt tính attention/gradient của TextGuider và
lượt tính `noise_pred` thật dùng **chính xác cùng một giá trị timestep**.

## 8. Base model — 50 bước, không phải variant distill

README chính thức xác nhận rõ: *"Use Base (50-step) for fine-tuning,
LoRA training, and maximum flexibility"* và bảng model liệt kê
`FLUX.2 [klein] 4B Base` có `Step-distilled = ❌`. Vậy `num_inference_steps=50`,
`t_guide_ratio=0.25` (≈12-13 bước đầu) trong config cũ **không sai** —
giữ nguyên.

## 9. Những gì vẫn CHƯA verify được 100% (để lại cảnh báo rõ trong code)

- Số layer double/single-stream **cụ thể của riêng Klein 4B** (không phải
  [dev]) — không tìm được số chính xác qua tài liệu công khai. Code không
  còn hardcode số này; luôn đọc từ `transformer.config` lúc chạy.
- Tên class Attention Processor chính xác trong Diffusers cho Flux2 —
  dùng phát hiện theo cấu trúc (`to_added_qkv`) thay vì đoán tên.
- Format chính xác mà `Flux2KleinPipeline.encode_prompt` tokenize prompt
  (có áp một template tối giản nào đó hay tokenize hoàn toàn thô) — code
  mới **tự kiểm chứng bằng assertion độ dài** thay vì giả định im lặng.

Khi chạy thật trên server, nếu có bất kỳ `AssertionError`/cảnh báo strict-mode
nào từ các điểm trên, đó chính là tín hiệu cần bạn tự in `transformer.config`,
`pipe.encode_prompt.__doc__`/source thật để khóa nốt các con số còn lại —
thay vì để chương trình âm thầm chạy sai như bản cũ.
