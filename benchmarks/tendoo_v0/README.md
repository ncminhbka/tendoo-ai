# TendooBizEval-Vi v0 Benchmark Specification

## Overview
`tendoo_v0` is a two-track benchmark designed to evaluate generative AI models (Text-to-Image and Image-to-Image) for Vietnamese commercial graphics, ad posters, and SME product banners.

## Tracks
1. **T2I (Text-to-Image)**: 50 prompt cases across consumer electronics, F&B, beauty/fitness, services/education/finance, and travel/home/fashion.
2. **I2I (Image-to-Image / Product Placement)**: 50 product reference cases, balanced across five edit types: background replacement, lifestyle placement, key visual, packaging/logo preservation, and object removal/cleanup.

Each track covers six layout families and seven output sizes: `1024x1024`, `1024x1280`, `1280x720`, `1024x1536`, `1024x768`, `1080x1920`, and `1200x628`. Cases also record text length, text block count, text types, language mix, and numeric pattern so OCR results can be sliced by difficulty.

## Structure
- `schema/`: JSON Schema definitions for test cases and evaluation results.
- `cases/`: `t2i.jsonl` and `i2i.jsonl` dataset files.
- `references/`: Product reference image storage.
- `outputs/`: Benchmark execution output images.
- `reports/`: Evaluated metrics in CSV and Markdown formats.
- `manifests/`: Manifest of reference images and status tracking.

## Reference images

Place product images under the path declared by `reference_image`. The current v0 manifest intentionally marks the 50 product files as pending until the reference set is supplied. A missing reference produces `PENDING_REFERENCE` during execution and is not silently scored as a generated result.

## Validation

Run from the repository root:

```bash
python src/benchmark/validate_dataset.py
```

The validator checks JSONL integrity, exact required-text presence, supported dimensions, aspect-ratio declarations, duplicate prompts, placeholder categories, category balance, and the five-way I2I edit distribution.
