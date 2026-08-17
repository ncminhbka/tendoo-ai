"""
Script kiểm tra tính hợp lệ của Benchmark Dataset (t2i.jsonl & i2i.jsonl)
"""

import json
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def validate_dataset():
    base_dir = "benchmarks/tendoo_v0/cases"
    t2i_path = os.path.join(base_dir, "t2i.jsonl")
    i2i_path = os.path.join(base_dir, "i2i.jsonl")
    
    total_cases = 0
    valid_cases = 0
    pending_refs = 0
    
    print("🔍 BẮT ĐẦU KIỂM TRA BENCHMARK DATASET...")
    
    for path, track in [(t2i_path, "t2i"), (i2i_path, "i2i")]:
        if not os.path.exists(path):
            print(f"❌ Không tìm thấy file: {path}")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip(): continue
                total_cases += 1
                try:
                    case = json.loads(line)
                    case_id = case.get("case_id")
                    instruction = case.get("instruction")
                    req_texts = case.get("required_text", [])
                    ref_img = case.get("reference_image")
                    
                    if not case_id or not instruction:
                        print(f"⚠️ [{track}:{line_idx}] Case thiếu ID hoặc instruction!")
                        continue
                        
                    if track == "i2i" and ref_img:
                        if not os.path.exists(ref_img):
                            pending_refs += 1
                            # Ghi nhận trạng thái pending_reference
                    valid_cases += 1
                except json.JSONDecodeError:
                    print(f"❌ [{track}:{line_idx}] Lỗi định dạng JSON!")

    print("-" * 60)
    print(f"📊 KẾT QUẢ VALIDATE BENCHMARK DATASET:")
    print(f"   • Tổng số cases: {total_cases}")
    print(f"   • Cases hợp lệ: {valid_cases}")
    print(f"   • Reference images chưa tải (pending_reference): {pending_refs}")
    print("-" * 60)
    
    return valid_cases == total_cases

if __name__ == "__main__":
    validate_dataset()
