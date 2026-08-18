"""DINOv2-based product identity preservation metric for I2I."""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_dinov2 = None
_processor = None
_failed = False


def _unavailable(reason: str) -> dict:
    return {"score": None, "status": "unavailable", "method": None, "reason": reason}


def get_dinov2_encoder():
    global _dinov2, _processor, _failed
    if _dinov2 is None and not _failed:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            model_id = os.getenv("TENDOO_DINOV2_MODEL", "facebook/dinov2-large")
            _processor = AutoImageProcessor.from_pretrained(model_id)
            _dinov2 = AutoModel.from_pretrained(model_id)
            _dinov2.to("cuda:0" if torch.cuda.is_available() else "cpu").eval()
        except Exception as exc:
            _failed = True
            print(f"[benchmark] DINOv2 unavailable: {exc}")
    return _dinov2, _processor


def evaluate_image_similarity(ref_image_path: str, output_image_path: str, return_details: bool = False):
    if not ref_image_path or not os.path.exists(ref_image_path):
        result = _unavailable("reference image does not exist")
    elif not output_image_path or not os.path.exists(output_image_path):
        result = _unavailable("output image does not exist")
    else:
        model, processor = get_dinov2_encoder()
        if model is None:
            result = _unavailable("DINOv2 is not installed or failed to load")
        else:
            try:
                import torch
                import torch.nn.functional as F
                from PIL import Image

                images = [Image.open(ref_image_path).convert("RGB"),
                          Image.open(output_image_path).convert("RGB")]
                inputs = processor(images=images, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs).last_hidden_state[:, 0]
                    outputs = F.normalize(outputs, p=2, dim=-1)
                    similarity = float(torch.sum(outputs[0] * outputs[1]).item())
                result = {"score": round(max(0.0, min(1.0, similarity)), 4),
                          "status": "measured", "method": "DINOv2-CLS-cosine", "reason": None}
            except Exception as exc:
                result = _unavailable(f"DINOv2 failed: {exc}")
    return result if return_details else result["score"]


if __name__ == "__main__":
    print("Image editing benchmark metric ready: DINOv2.")
