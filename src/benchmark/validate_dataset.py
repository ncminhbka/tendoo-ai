"""Validate benchmark JSONL content, coverage and reference readiness for FLUX.2 Specs."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = ROOT / "benchmarks" / "tendoo_v0" / "cases"

# FLUX.2 Klein strict multiple of 16 resolutions
ALLOWED_SIZES = {
    (1024, 1024),
    (1024, 1280),
    (1280, 720),
    (1024, 1536),
    (1024, 768),
    (1088, 1920),
    (1200, 624)
}

ALLOWED_EDITS = {
    "background_replacement",
    "lifestyle_placement",
    "key_visual",
    "preserve_packaging_logo",
    "object_removal_cleanup"
}

PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]|placeholder_\d+", re.I)

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows

def validate_dataset(verbose: bool = True) -> bool:
    errors, all_rows = [], []
    for track in ("t2i", "i2i"):
        path = CASE_DIR / f"{track}.jsonl"
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        try:
            rows = read_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if len(rows) != 50:
            errors.append(f"{track}: expected 50 cases, got {len(rows)}")

        all_rows.extend(rows)
        ids = [row.get("case_id") for row in rows]

        if len(ids) != len(set(ids)):
            errors.append(f"{track}: duplicate case_id")

        for row in rows:
            cid, instruction, texts = row.get("case_id", "?"), row.get("instruction", ""), row.get("required_text", [])
            if row.get("track") != track:
                errors.append(f"{cid}: wrong track")

            if not instruction or not texts:
                errors.append(f"{cid}: missing instruction or required_text")

            for target in texts:
                if target not in instruction:
                    errors.append(f"{cid}: required text not found in prompt: {target}")

            if PLACEHOLDER_RE.search(instruction):
                errors.append(f"{cid}: placeholder text in instruction")

            size = tuple(row.get("output_size", []))
            if size not in ALLOWED_SIZES:
                errors.append(f"{cid}: unsupported output_size {size} for FLUX.2")

            # Kiểm tra bội số 16
            if len(size) == 2:
                if size[0] % 16 != 0 or size[1] % 16 != 0:
                    errors.append(f"{cid}: output_size {size} not divisible by 16")

            if track == "i2i":
                ref = row.get("reference_image")
                ref_full_path = ROOT / ref if ref else None
                if not ref or not ref_full_path.exists():
                    errors.append(f"{cid}: missing or non-existent reference image path: {ref}")
                if row.get("edit_type") not in ALLOWED_EDITS:
                    errors.append(f"{cid}: invalid edit_type {row.get('edit_type')}")

    for track in ("t2i", "i2i"):
        rows = [row for row in all_rows if row.get("track") == track]
        for field in ("target_layout", "difficulty", "text_length", "output_size"):
            if len({str(row.get(field)) for row in rows}) < 3:
                errors.append(f"{track}: insufficient diversity in {field}")

        if any(count > 1 for count in Counter(row.get("instruction", "") for row in rows).values()):
            errors.append(f"{track}: duplicate instructions")

    edits = Counter(row.get("edit_type") for row in all_rows if row.get("track") == "i2i")
    if set(edits) != ALLOWED_EDITS or any(value != 10 for value in edits.values()):
        errors.append(f"i2i: edit distribution is {dict(edits)}")

    if verbose:
        print(f"cases={len(all_rows)} errors={len(errors)}")
        for error in errors[:20]:
            print(f"ERROR: {error}")
        for track in ("t2i", "i2i"):
            rows = [row for row in all_rows if row.get("track") == track]
            print(f"{track}: layouts={len({row.get('target_layout') for row in rows})}, sizes={len({tuple(row.get('output_size', [])) for row in rows})}, difficulties={dict(Counter(row.get('difficulty') for row in rows))}")

    return not errors

if __name__ == "__main__":
    raise SystemExit(0 if validate_dataset() else 1)
