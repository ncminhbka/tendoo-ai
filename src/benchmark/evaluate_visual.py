"""Reference-backed visual metrics for T2I evaluation.

VQAScore is used for prompt-image alignment and the LAION aesthetic predictor
is used for image quality. Failed optional models are reported as unavailable;
there is deliberately no numeric heuristic fallback.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_vqa_scorer = None
_vqa_failed = False
_aesthetic_model = None
_aesthetic_failed = False


def _unavailable(reason: str) -> dict:
    return {"score": None, "status": "unavailable", "method": None, "reason": reason}


def get_vqascore_evaluator():
    global _vqa_scorer, _vqa_failed
    if _vqa_scorer is None and not _vqa_failed:
        try:
            import torch
            import t2v_metrics

            model = os.getenv("TENDOO_VQASCORE_MODEL", "clip-flant5-xl")
            _vqa_scorer = t2v_metrics.VQAScore(
                model=model,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except Exception as exc:
            _vqa_failed = True
            print(f"[benchmark] VQAScore unavailable: {exc}")
    return _vqa_scorer


def evaluate_prompt_alignment(prompt: str, image_path: str, return_details: bool = False):
    if not image_path or not os.path.exists(image_path):
        result = _unavailable("output image does not exist")
    else:
        scorer = get_vqascore_evaluator()
        if scorer is None:
            result = _unavailable("t2v_metrics/VQAScore is not installed or failed to load")
        else:
            try:
                raw = scorer(images=[image_path], texts=[prompt])
                score = float(raw[0][0] if getattr(raw, "ndim", 1) == 2 else raw[0])
                result = {"score": round(max(0.0, min(1.0, score)), 4),
                          "status": "measured", "method": "VQAScore", "reason": None}
            except Exception as exc:
                result = _unavailable(f"VQAScore failed: {exc}")
    return result if return_details else result["score"]


def get_aesthetic_evaluator():
    """Load LAION's published linear aesthetic predictor on CLIP embeddings."""
    global _aesthetic_model, _aesthetic_failed
    if _aesthetic_model is None and not _aesthetic_failed:
        try:
            import torch
            import open_clip

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            predictor_path = os.getenv("TENDOO_AESTHETIC_WEIGHTS")
            if not predictor_path or not os.path.exists(predictor_path):
                raise RuntimeError("set TENDOO_AESTHETIC_WEIGHTS to the LAION predictor .pth file")
            predictor = torch.nn.Linear(768, 1)
            predictor.load_state_dict(torch.load(predictor_path, map_location="cpu"))
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            _aesthetic_model = (model.to(device).eval(), preprocess, predictor.to(device).eval(), device)
        except Exception as exc:
            _aesthetic_failed = True
            print(f"[benchmark] LAION aesthetic predictor unavailable: {exc}")
    return _aesthetic_model


def evaluate_aesthetic_score(image_path: str, return_details: bool = False):
    if not image_path or not os.path.exists(image_path):
        result = _unavailable("output image does not exist")
    else:
        evaluator = get_aesthetic_evaluator()
        if evaluator is None:
            result = _unavailable("LAION aesthetic predictor is not installed/configured")
        else:
            try:
                import torch
                from PIL import Image

                model, preprocess, predictor, device = evaluator
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = model.encode_image(image)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    raw = float(predictor(embedding.float()).item())
                result = {"score": round(max(0.0, min(1.0, (raw - 1.0) / 9.0)), 4),
                          "raw_score": round(raw, 4), "status": "measured",
                          "method": "LAION-Aesthetics-Predictor", "reason": None}
            except Exception as exc:
                result = _unavailable(f"LAION aesthetic predictor failed: {exc}")
    return result if return_details else result["score"]


if __name__ == "__main__":
    print("Visual benchmark metrics ready: VQAScore + LAION-Aesthetics.")
