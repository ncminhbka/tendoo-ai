"""Report generation for the TendooBizEval-Vi benchmark.

The report only aggregates metrics that were actually measured. Missing
optional evaluators are represented as null/unavailable and never become a
synthetic score.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from math import ceil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TRACK_WEIGHTS = {
    "t2i": {
        "text": 0.40,
        "alignment": 0.30,
        "aesthetic": 0.20,
    },
    "i2i": {
        "text": 0.25,
        "alignment": 0.20,
        "aesthetic": 0.15,
        "preservation": 0.40,
    },
}

METRIC_LABELS = {
    "text": "Text accuracy",
    "alignment": "Prompt alignment",
    "aesthetic": "Aesthetic quality",
    "preservation": "Product preservation",
}


def _metric_values(track: str, metrics: dict) -> dict:
    provenance = metrics.get("metric_provenance", {})
    ocr_status = provenance.get("ocr", {}).get("status")
    text_available = ocr_status == "measured"
    ned = metrics.get("ned")
    exact = metrics.get("exact_match_ratio")

    return {
        "text": ((ned * 0.7 + exact * 0.3) if text_available and
                 isinstance(ned, (int, float)) and isinstance(exact, (int, float)) else None),
        "alignment": metrics.get("prompt_alignment") if provenance.get("alignment", {}).get("status") == "measured" else None,
        "aesthetic": metrics.get("aesthetic_score") if provenance.get("aesthetic", {}).get("status") == "measured" else None,
        "preservation": metrics.get("product_similarity") if track == "i2i" and provenance.get("product_similarity", {}).get("status") == "measured" else None,
    }


def calculate_score_details(track: str, metrics: dict) -> dict:
    """Return score, coverage and the measured dimensions for one run."""
    weights = TRACK_WEIGHTS.get(track, {})
    values = _metric_values(track, metrics)
    valid = {name: value for name, value in values.items()
             if name in weights and isinstance(value, (int, float))}
    coverage = sum(weights[name] for name in valid)
    score = (sum(valid[name] * weights[name] for name in valid) / coverage * 100.0
             if coverage else 0.0)

    cer = metrics.get("cer")
    if "text" in valid and isinstance(cer, (int, float)) and cer > 0.30:
        score = min(score, 30.0)

    return {
        "score": round(score, 2),
        "coverage": round(coverage, 2),
        "available_metrics": len(valid),
        "expected_metrics": len(weights),
        "measured_metrics": list(valid.keys()),
    }


def calculate_composite_score(track: str, metrics: dict) -> float:
    """Backward-compatible score accessor."""
    return calculate_score_details(track, metrics)["score"]


def _mean(values):
    return round(sum(values) / len(values), 2) if values else None


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, ceil(len(ordered) * 0.95) - 1)], 2)


def _status(metrics: dict, name: str) -> str:
    provenance = metrics.get("metric_provenance", {})
    key = {"text": "ocr", "alignment": "alignment", "aesthetic": "aesthetic",
           "preservation": "product_similarity"}.get(name)
    if key and key in provenance:
        status = provenance[key].get("status", "unavailable")
        if status.startswith("unavailable"):
            return "unavailable"
        if status in ("measured", "proxy"):
            return status
        return "unavailable"
    return "legacy"


def _method(metrics: dict, name: str) -> str:
    provenance = metrics.get("metric_provenance", {})
    key = {"text": "ocr", "alignment": "alignment", "aesthetic": "aesthetic",
           "preservation": "product_similarity"}.get(name)
    return str(provenance.get(key, {}).get("method") or "") if key else ""


def generate_reports(results: list, output_dir: str = "benchmarks/tendoo_v1/reports"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "report.csv")
    md_path = os.path.join(output_dir, "report.md")

    case_metadata = {}
    for track in ("t2i", "i2i"):
        case_path = os.path.join("benchmarks", "tendoo_v1", "cases", f"{track}.jsonl")
        if os.path.exists(case_path):
            with open(case_path, encoding="utf-8") as case_file:
                for line in case_file:
                    if line.strip():
                        case = json.loads(line)
                        case_metadata[case["case_id"]] = case

    total_runs = len(results)
    pending_runs = sum(r.get("status") == "PENDING_REFERENCE" for r in results)
    completed = [r for r in results if r.get("status") != "PENDING_REFERENCE"]
    pass_runs = sum(r.get("status") == "PASS" for r in completed)
    fail_text_runs = sum(r.get("status") == "FAIL_TEXT" for r in results)
    fail_identity_runs = sum(r.get("status") == "FAIL_IDENTITY" for r in results)
    technical_runs = sum(r.get("status") == "FAIL_TECHNICAL" for r in results)

    score_by_track = defaultdict(list)
    coverage_by_track = defaultdict(list)
    latency_values = []
    vram_values = []
    grouped = defaultdict(lambda: {"runs": 0, "scores": [], "coverage": [], "latency": []})
    metric_counts = defaultdict(lambda: {"measured": 0, "unavailable": 0, "proxy": 0, "legacy": 0})

    csv_fields = [
        "case_id", "track", "seed", "status", "category", "layout", "difficulty",
        "output_size", "text_length", "cer", "wer", "ned", "exact_match_ratio",
        "product_similarity", "prompt_alignment", "aesthetic_score", "score_coverage",
        "composite_score", "latency_s", "peak_vram_gb", "ocr_status", "ocr_method",
        "alignment_status", "alignment_method", "aesthetic_status", "aesthetic_method",
        "preservation_status", "preservation_method",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            cid, track = result["case_id"], result["track"]
            metrics = result.get("metrics", {})
            details = calculate_score_details(track, metrics)
            score = metrics.get("composite_score", details["score"])
            coverage = metrics.get("score_coverage", details["coverage"])
            case = case_metadata.get(cid, {})
            attrs = case.get("product_attributes", {})
            provenance = metrics.get("metric_provenance", {})
            latency = result.get("latency_seconds")
            vram = result.get("peak_vram_gb")

            row = {
                "case_id": cid, "track": track, "seed": result.get("seed"),
                "status": result.get("status"), "category": attrs.get("category", "unknown"),
                "layout": case.get("target_layout", "unknown"),
                "difficulty": case.get("difficulty", "unknown"),
                "output_size": "x".join(map(str, case.get("output_size", []))),
                "text_length": case.get("text_length", "unknown"),
                "cer": metrics.get("cer"), "wer": metrics.get("wer"),
                "ned": metrics.get("ned"), "exact_match_ratio": metrics.get("exact_match_ratio"),
                "product_similarity": metrics.get("product_similarity"),
                "prompt_alignment": metrics.get("prompt_alignment"),
                "aesthetic_score": metrics.get("aesthetic_score"),
                "score_coverage": coverage, "composite_score": score,
                "latency_s": latency, "peak_vram_gb": vram,
                "ocr_status": provenance.get("ocr", {}).get("status"),
                "ocr_method": provenance.get("ocr", {}).get("method"),
                "alignment_status": provenance.get("alignment", {}).get("status"),
                "alignment_method": provenance.get("alignment", {}).get("method"),
                "aesthetic_status": provenance.get("aesthetic", {}).get("status"),
                "aesthetic_method": provenance.get("aesthetic", {}).get("method"),
                "preservation_status": provenance.get("product_similarity", {}).get("status") if provenance.get("product_similarity") else None,
                "preservation_method": provenance.get("product_similarity", {}).get("method") if provenance.get("product_similarity") else None,
            }
            writer.writerow(row)

            if coverage > 0 and result.get("status") != "PENDING_REFERENCE":
                score_by_track[track].append(score)
                coverage_by_track[track].append(coverage)
            if isinstance(latency, (int, float)) and latency > 0:
                latency_values.append(latency)
            if isinstance(vram, (int, float)) and vram > 0:
                vram_values.append(vram)
            for metric_name in TRACK_WEIGHTS.get(track, {}):
                metric_status = _status(metrics, metric_name)
                metric_counts[(track, metric_name)][metric_status] += 1

            for dimension, value in (
                ("category", attrs.get("category", "unknown")),
                ("layout", case.get("target_layout", "unknown")),
                ("difficulty", case.get("difficulty", "unknown")),
                ("output_size", "x".join(map(str, case.get("output_size", [])))),
                ("text_length", case.get("text_length", "unknown")),
            ):
                bucket = grouped[(track, dimension, value)]
                bucket["runs"] += 1
                if coverage > 0:
                    bucket["scores"].append(score)
                    bucket["coverage"].append(coverage)
                if isinstance(latency, (int, float)) and latency > 0:
                    bucket["latency"].append(latency)

    dataset_counts = {track: sum(case.get("track") == track for case in case_metadata.values())
                      for track in ("t2i", "i2i")}
    result_case_counts = {track: len({r.get("case_id") for r in results if r.get("track") == track})
                          for track in ("t2i", "i2i")}
    stale_dataset = any(result_case_counts[t] < dataset_counts[t] for t in dataset_counts)
    avg_t2i, avg_i2i = _mean(score_by_track["t2i"]), _mean(score_by_track["i2i"])
    avg_coverage = _mean([r.get("metrics", {}).get("score_coverage", 0.0) for r in results])
    avg_latency = _mean(latency_values)
    p95_latency = _p95(latency_values)
    min_latency = round(min(latency_values), 2) if latency_values else None
    max_latency = round(max(latency_values), 2) if latency_values else None
    throughput = round(60.0 / avg_latency, 1) if avg_latency else None
    avg_vram = _mean(vram_values)
    pass_rate_completed = round(pass_runs / len(completed) * 100, 1) if completed else None
    pass_rate_all = round(pass_runs / total_runs * 100, 1) if total_runs else None

    breakdown_rows = []
    for (track, dimension, value), bucket in sorted(grouped.items()):
        score_average = _mean(bucket["scores"])
        coverage_average = _mean(bucket["coverage"])
        latency_average = _mean(bucket["latency"])
        breakdown_rows.append(
            f"| {track} | {dimension} | {value} | {bucket['runs']} | "
            f"{score_average if score_average is not None else '-'} | "
            f"{coverage_average if coverage_average is not None else '-'} | "
            f"{latency_average if latency_average is not None else '-'}s |"
        )
    breakdown_table = "\n".join(breakdown_rows) or "| - | - | - | 0 | - | - | - |"

    def score_text(value):
        return f"{value:.2f} / 100" if isinstance(value, (int, float)) else "N/A"

    def pct_text(value):
        return f"{value:.1f}%" if isinstance(value, (int, float)) else "N/A"

    t2i_weights = ", ".join(f"{METRIC_LABELS[k]} {v:.0%}" for k, v in TRACK_WEIGHTS["t2i"].items())
    i2i_weights = ", ".join(f"{METRIC_LABELS[k]} {v:.0%}" for k, v in TRACK_WEIGHTS["i2i"].items())
    metric_rows = []
    for track, weights in TRACK_WEIGHTS.items():
        for name, weight in weights.items():
            counts = metric_counts[(track, name)]
            metric_rows.append(f"| {track} | {METRIC_LABELS[name]} | {weight:.0%} | {counts['measured']} | {counts['unavailable']} | {counts['proxy']} | {counts['legacy']} |")
    metric_table = "\n".join(metric_rows) or "| - | - | - | 0 | 0 | 0 | 0 |"

    md_content = f"""# TendooBizEval-Vi Benchmark Report

Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`
Runs: `{total_runs}`
Dataset: `{dataset_counts['t2i']} T2I + {dataset_counts['i2i']} I2I`
Dataset status: `{'STALE - rerun missing cases' if stale_dataset else 'COMPLETE'}`

## Executive Summary

| Metric | Value | Interpretation |
| :--- | ---: | :--- |
| Pass rate among completed runs | `{pct_text(pass_rate_completed)}` | Excludes pending reference runs |
| Pass rate across all runs | `{pct_text(pass_rate_all)}` | Includes pending reference runs |
| T2I composite score | `{score_text(avg_t2i)}` | Average of runs with coverage > 0 |
| I2I composite score | `{score_text(avg_i2i)}` | Average of runs with coverage > 0 |
| Average score coverage | `{pct_text(avg_coverage * 100 if avg_coverage is not None else None)}` | Share of configured weight measured |
| FAIL_TEXT | `{fail_text_runs}` | OCR/CER gate failed or OCR unavailable |
| FAIL_IDENTITY | `{fail_identity_runs}` | DINOv2 identity gate failed |
| FAIL_TECHNICAL | `{technical_runs}` | Generation/runtime failure |
| PENDING_REFERENCE | `{pending_runs}` | Reference image not available |

## Composite Score

Scores are normalized over available dimensions. Missing or unavailable metrics are excluded from both numerator and denominator; they are never replaced with a default score.

T2I weights: `{t2i_weights}`.
I2I weights: `{i2i_weights}`.

Text accuracy is `0.7 * NED + 0.3 * Exact Match Ratio`. A measured CER above 30% caps the run score at 30. A score with zero coverage is reported as `0` in the machine-readable output but is marked unusable for track averages.

## Metric Availability

| Track | Metric | Configured weight | Measured | Unavailable | Proxy | Legacy/unknown |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
{metric_table}

## Runtime

| Metric | Value |
| :--- | ---: |
| Mean latency | `{avg_latency}s` |
| P95 latency | `{p95_latency}s` |
| Min / max latency | `{min_latency}s / {max_latency}s` |
| Throughput | `{throughput} images/min` |
| Average peak VRAM | `{f'{avg_vram} GB' if avg_vram is not None else 'N/A'}` |

## Breakdown By Case Metadata

| Track | Dimension | Value | Runs | Avg Score | Avg Coverage | Avg Latency |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
{breakdown_table}

## Output Files

- `report.csv`: one row per run, including raw metrics, availability status, methods, score coverage and composite score.
- `result.jsonl`: full run records, including metric provenance and generated image paths.
"""
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(md_content)

    print("Benchmark reports exported:")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    print("Report module ready.")
