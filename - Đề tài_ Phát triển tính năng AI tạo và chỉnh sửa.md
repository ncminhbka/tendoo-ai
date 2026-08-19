## 🏆 Top 3 Paper đề xuất (theo thứ tự ưu tiên)

### 1. **EasyText** (arXiv:2505.24417) — Lựa chọn số 1 cho FLUX.2 [dev]

**Link:** https://arxiv.org/pdf/2505.24417

**Tại sao chọn:**

- **Kiến trúc phù hợp:** Dựa trên **Diffusion Transformer (DiT)** — trùng khớp với kiến trúc của FLUX.2 [dev] (cũng là DiT-based).[^1]
- **Multilingual support:** Đã test trên **tiếng Việt** với kết quả tốt (0.9605 accuracy) — đúng target của bạn.[^1]
- **Đơn giản implement:** Chỉ cần thêm **character token encoding** + **position encoding interpolation** vào pipeline hiện có.[^1]
- **Data requirement:** Có sẵn dataset 1M synthetic + 20K high-quality annotated — có thể fine-tune ngay.[^1]
- **Hardware:** Fine-tune được trên 1-2 GPU A100/H100 (hoặc 4-8 GPU consumer-grade với gradient checkpointing).

**Ý tưởng áp dụng:**

- Mã hóa từng ký tự tiếng Việt (có dấu) thành token riêng.
- Dùng positional encoding để kiểm soát vị trí text trong ảnh.
- Fine-tune FLUX.2 [dev] trên dataset tiếng Việt (poster, banner, menu...).

***

### 2. **FreeText** (arXiv:2601.00535) — Training-free, plug-and-play

**Link:** https://arxiv.org/html/2601.00535v1

**Tại sao chọn:**

- **Không cần fine-tune:** Hoạt động ở inference time, không sửa model weights.[^2]
- **Base-model agnostic:** Đã test thành công trên **FLUX.1-dev**, SD3, Qwen-Image.[^2]
- **Cải thiện đáng kể:** Tăng readability text mà không làm giảm chất lượng ảnh.[^2]
- **Hardware:** Không yêu cầu thêm GPU — chỉ tăng ~10-20% inference time.

**Ý tưởng áp dụng:**

- Dùng làm **baseline nhanh** để test khả năng sinh text tiếng Việt của FLUX.2 [dev] trước khi fine-tune.
- Có thể kết hợp với EasyText sau khi fine-tune để push performance thêm.

***

### 3. **TextGuider** (arXiv:2512.09350) — Giải quyết text omission

**Link:** https://arxiv.org/abs/2512.09350v2

**Tại sao chọn:**

- **Giải quyết vấn đề thực tế:** Text bị thiếu/mất ký tự (omission) — lỗi phổ biến khi sinh text tiếng Việt có dấu.[^3]
- **Training-free:** Cũng là inference-time guidance, dễ tích hợp.[^3]
- **MM-DiT compatible:** Phân tích attention patterns trong Multi-Modal DiT — phù hợp FLUX.2.[^3]
- **State-of-the-art:** Đạt recall cao nhất trong các benchmark text rendering.[^3]

**Ý tưởng áp dụng:**

- Dùng latent guidance ở early denoising steps để ensure text xuất hiện đầy đủ.
- Kết hợp với EasyText fine-tune để vừa accurate vừa complete.

***

## 📊 Bảng so sánh nhanh

| Paper | Kiến trúc | Fine-tune | Data cần | Hardware | Phù hợp FLUX.2 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **EasyText** | DiT | ✅ Có | 1M synthetic + 20K HQ | 1-2 A100/H100 | ⭐⭐⭐⭐⭐ (trùng DiT) |
| **FreeText** | Any (plug-in) | ❌ Không | Không cần | CPU/GPU bất kỳ | ⭐⭐⭐⭐ (test nhanh) |
| **TextGuider** | MM-DiT | ❌ Không | Không cần | CPU/GPU bất kỳ | ⭐⭐⭐⭐ (fix omission) |


***

## 🚀 Lộ trình implement đề xuất

### Phase 1: Baseline (1-2 tuần)

1. **Setup FLUX.2 [dev] base** — test sinh text tiếng Việt cơ bản (không fine-tune).
2. **Áp dụng FreeText** — đo cải thiện OCR accuracy, recall.[^2]
3. **Áp dụng TextGuider** — đo cải thiện text completeness (giảm omission).[^3]

### Phase 2: Fine-tune với EasyText (3-4 tuần)

1. **Chuẩn bị data:**
    - Thu thập ~10K-50K ảnh poster/banner/menu tiếng Việt (có text rõ).
    - Dùng OCR (PaddleOCR/VietOCR) để extract text + bounding box.
    - Augment data (synthetic text rendering) nếu cần.
2. **Implement EasyText modules:**
    - Character token encoder (Unicode tiếng Việt).
    - Position encoding interpolation.
    - Loss function cho text rendering (OCR loss + position loss).
3. **Fine-tune FLUX.2 [dev]:**
    - Freeze image backbone, chỉ train text rendering heads.
    - Dùng gradient checkpointing + LoRA để giảm memory.
4. **Eval:**
    - OCR accuracy (tiếng Việt có dấu).
    - Aesthetic quality (CLIP score, human eval).
    - Inference speed.

### Phase 3: Optimization (2-3 tuần)

1. **Combine EasyText + TextGuider** — fine-tune model + inference-time guidance.
2. **Test trên các case khó:**
    - Text dài (banner khai trương).
    - Text nhiều dòng (menu món).
    - Text cong/uốn (logo, sticker).
3. **Deploy:**
    - Quantize model (INT8/FP16) để giảm inference time.
    - Build API endpoint cho user upload prompt + image.

***

## 📚 Paper bổ sung nên đọc (nếu có thời gian)

- **TextCrafter** (arXiv:2503.23461) — Rendering multiple texts trong complex scenes.[^4]
- **TextDiffuser-2** — Dùng LLM để improve text rendering (có thể áp dụng cho prompt engineering tiếng Việt).
- **WordCon** (arXiv:2506.21276) — Word-level typography control (nếu cần fine control font/size/color).

***

## 💡 Gợi ý data tiếng Việt

- **Dataset công khai:**
    - VTDR (Vietnamese Text Detection and Recognition)
    - ICDAR 2019 ArT (có text tiếng Việt)
- **Self-collect:**
    - Crawl ảnh poster/banner từ Facebook, Shopee, Tiki.
    - Dùng OCR để label tự động.
- **Synthetic:**
    - Dùng EasyText's 1M synthetic dataset làm base, augment thêm tiếng Việt.

***

Nếu bạn cần, tôi có thể:

1. **Fetch full PDF** của EasyText/FreeText/TextGuider để extract chi tiết implementation.
2. **Viết code mẫu** cho character token encoder + position encoding (Python/PyTorch).
3. **Search dataset** tiếng Việt cụ thể cho text rendering.

Bạn muốn bắt đầu từ đâu?
<span style="display:none">[^10][^11][^12][^13][^14][^15][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/html/2505.24417v2

[^2]: https://arxiv.org/html/2601.00535v1

[^3]: https://arxiv.org/abs/2512.09350v2

[^4]: https://aclanthology.org/2025.emnlp-main.1070.pdf

[^5]: https://pubmed.ncbi.nlm.nih.gov/41418008/

[^6]: https://www.computer.org/csdl/journal/tp/2025/09/11002717/26GmRnP6FFe

[^7]: https://web3.arxiv.org/abs/2511.22699

[^8]: https://arxiv.org/abs/2602.09268

[^9]: https://discovery.ucl.ac.uk/id/eprint/10217177/

[^10]: https://arxiv.org/html/2601.16208v1

[^11]: https://huggingface.co/papers/2512.18254

[^12]: https://huggingface.co/papers/2605.10730

[^13]: https://huggingface.co/papers/2605.07748

[^14]: https://www.scribd.com/document/910715287/2503-07703v1

[^15]: https://chatpaper.com/chatpaper/paper/143610

