# TendooBizEval-Vi v0 Benchmark Specification

## Overview
`tendoo_v0` is a two-track benchmark designed to evaluate generative AI models (Text-to-Image and Image-to-Image) for Vietnamese commercial graphics, ad posters, and SME product banners.

## Tracks
1. **T2I (Text-to-Image)**: 50 prompt cases converting Vietnamese business prompts into commercial posters.
2. **I2I (Image-to-Image / Product Placement)**: 50 product reference cases evaluating background replacement, lifestyle placement, and branding preservation.

## Structure
- `schema/`: JSON Schema definitions for test cases and evaluation results.
- `cases/`: `t2i.jsonl` and `i2i.jsonl` dataset files.
- `references/`: Product reference image storage.
- `outputs/`: Benchmark execution output images.
- `reports/`: Evaluated metrics in CSV and Markdown formats.
- `manifests/`: Manifest of reference images and status tracking.
