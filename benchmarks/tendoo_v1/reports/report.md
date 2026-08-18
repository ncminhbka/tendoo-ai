# TendooBizEval-Vi Benchmark Report

Generated: `2026-08-18T17:10:19+07:00`
Runs: `6`
Dataset: `50 T2I + 46 I2I`
Dataset status: `STALE - rerun missing cases`

## Executive Summary

| Metric | Value | Interpretation |
| :--- | ---: | :--- |
| Pass rate among completed runs | `0.0%` | Excludes pending reference runs |
| Pass rate across all runs | `0.0%` | Includes pending reference runs |
| T2I composite score | `N/A` | Average of runs with coverage > 0 |
| I2I composite score | `N/A` | Average of runs with coverage > 0 |
| Average score coverage | `0.0%` | Share of configured weight measured |
| FAIL_TEXT | `6` | OCR/CER gate failed or OCR unavailable |
| FAIL_IDENTITY | `0` | DINOv2 identity gate failed |
| FAIL_TECHNICAL | `0` | Generation/runtime failure |
| PENDING_REFERENCE | `0` | Reference image not available |

## Composite Score

Scores are normalized over available dimensions. Missing or unavailable metrics are excluded from both numerator and denominator; they are never replaced with a default score.

T2I weights: `Text accuracy 40%, Prompt alignment 30%, Aesthetic quality 20%`.
I2I weights: `Text accuracy 25%, Prompt alignment 20%, Aesthetic quality 15%, Product preservation 40%`.

Text accuracy is `0.7 * NED + 0.3 * Exact Match Ratio`. A measured CER above 30% caps the run score at 30. A score with zero coverage is reported as `0` in the machine-readable output but is marked unusable for track averages.

## Metric Availability

| Track | Metric | Configured weight | Measured | Unavailable | Proxy | Legacy/unknown |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| t2i | Text accuracy | 40% | 0 | 0 | 0 | 6 |
| t2i | Prompt alignment | 30% | 0 | 0 | 0 | 6 |
| t2i | Aesthetic quality | 20% | 0 | 0 | 0 | 6 |
| i2i | Text accuracy | 25% | 0 | 0 | 0 | 0 |
| i2i | Prompt alignment | 20% | 0 | 0 | 0 | 0 |
| i2i | Aesthetic quality | 15% | 0 | 0 | 0 | 0 |
| i2i | Product preservation | 40% | 0 | 0 | 0 | 0 |

## Runtime

| Metric | Value |
| :--- | ---: |
| Mean latency | `0.77s` |
| P95 latency | `2.13s` |
| Min / max latency | `0.5s / 2.13s` |
| Throughput | `77.9 images/min` |
| Average peak VRAM | `N/A` |

## Breakdown By Case Metadata

| Track | Dimension | Value | Runs | Avg Score | Avg Coverage | Avg Latency |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| t2i | category | unknown | 6 | - | - | 0.77s |
| t2i | difficulty | easy | 3 | - | - | 1.04s |
| t2i | difficulty | medium | 3 | - | - | 0.5s |
| t2i | layout | square_1x1 | 3 | - | - | 1.04s |
| t2i | layout | vertical_4x5 | 3 | - | - | 0.5s |
| t2i | output_size | 1024x1024 | 3 | - | - | 1.04s |
| t2i | output_size | 1024x1280 | 3 | - | - | 0.5s |
| t2i | text_length | medium | 3 | - | - | 0.5s |
| t2i | text_length | short | 3 | - | - | 1.04s |

## Output Files

- `report.csv`: one row per run, including raw metrics, availability status, methods, score coverage and composite score.
- `result.jsonl`: full run records, including metric provenance and generated image paths.
