"""
Script kiểm tra toàn bộ cấu hình Phần cứng (GPU, VRAM, CPU, RAM) và Môi trường Phần mềm (PyTorch, CUDA, Libraries)
Dự án: Tendoo Media AI - Viettel Telecom
"""

import os
import sys
import platform
import json
from datetime import datetime

def get_size_gb(bytes_val):
    return round(bytes_val / (1024 ** 3), 2)

def audit_environment():
    report = {}
    report["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Hệ thống & CPU
    report["system"] = {
        "os": f"{platform.system()} {platform.release()}",
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    
    # Tính RAM hệ thống nếu có psutil
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        report["system"]["ram_gb"] = ram_gb
    except ImportError:
        report["system"]["ram_gb"] = "N/A (Cần cài psutil)"

    # 2. PyTorch & CUDA
    pytorch_info = {}
    try:
        import torch
        pytorch_info["torch_version"] = torch.__version__
        pytorch_info["cuda_available"] = torch.cuda.is_available()
        pytorch_info["cuda_version"] = torch.version.cuda
        pytorch_info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else "N/A"
        
        if torch.cuda.is_available():
            pytorch_info["gpu_count"] = torch.cuda.device_count()
            pytorch_info["is_bf16_supported"] = torch.cuda.is_bf16_supported()
            
            gpus = []
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                total_vram = get_size_gb(prop.total_memory)
                gpus.append({
                    "id": i,
                    "name": prop.name,
                    "vram_gb": total_vram,
                    "compute_capability": f"{prop.major}.{prop.minor}"
                })
            pytorch_info["gpus"] = gpus
    except ImportError:
        pytorch_info["error"] = "PyTorch chưa được cài đặt"
    
    report["pytorch"] = pytorch_info

    # 3. Kiểm tra các thư viện AI chủ chốt
    packages_to_check = [
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "safetensors",
        "vietocr",
        "cv2",
        "PIL",
        "pandas",
        "numpy",
        "scipy",
        "einops",
        "sentencepiece"
    ]
    
    pkg_info = {}
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg if pkg != "cv2" else "cv2")
            version = getattr(mod, "__version__", "Đã cài (Không có __version__)")
            pkg_info[pkg] = version
        except ImportError:
            pkg_info[pkg] = "🔴 Chưa cài đặt"
            
    report["packages"] = pkg_info

    # 4. Hiển thị thông tin ra màn hình
    print("=" * 65)
    print("      📊 BÁO CÁO CẤU HÌNH PHẦN CỨNG & MÔI TRƯỜNG AI SERVER      ")
    print("=" * 65)
    print(f"⏰ Thời gian kiểm tra : {report['timestamp']}")
    print(f"💻 Hệ điều hành        : {report['system']['os']}")
    print(f"🐍 Python Version      : {report['system']['python_version']}")
    print(f"🧮 Số nhân CPU         : {report['system']['cpu_count']} cores")
    print(f"💾 Dung lượng RAM      : {report['system']['ram_gb']} GB")
    print("-" * 65)
    
    pt = report.get("pytorch", {})
    if pt.get("cuda_available"):
        print(f"🔥 PyTorch Version      : {pt.get('torch_version')}")
        print(f"⚡ CUDA Version         : {pt.get('cuda_version')}")
        print(f"🟢 Hỗ trợ bfloat16     : {pt.get('is_bf16_supported')}")
        print(f"🎮 Số lượng GPU         : {pt.get('gpu_count')} card(s)")
        for gpu in pt.get("gpus", []):
            print(f"   └─ [GPU {gpu['id']}] {gpu['name']} | VRAM: {gpu['vram_gb']} GB | Compute: {gpu['compute_capability']}")
    else:
        print("❌ PyTorch không nhận diện được GPU CUDA!")
        
    print("-" * 65)
    print("📦 PHIÊN BẢN CÁC THƯ VIỆN PHẦN MỀM CHỦ CHỐT:")
    for pkg, ver in report["packages"].items():
        status_icon = "✅" if "🔴" not in str(ver) else "❌"
        print(f"   {status_icon} {pkg:<20}: {ver}")
    print("=" * 65)

    # 5. Xuất ra tệp ENVIRONMENT_INFO.md & sys_info.json
    md_content = f"""# BÁO CÁO CẤU HÌNH HẠ TẦNG AI SERVER

**Thời gian kiểm tra**: {report['timestamp']}

## 1. Cấu hình Phần cứng & Hệ điều hành
- **Hệ điều hành**: `{report['system']['os']}`
- **Phiên bản Python**: `{report['system']['python_version']}`
- **CPU Cores**: `{report['system']['cpu_count']}` cores
- **Dung lượng RAM**: `{report['system']['ram_gb']}` GB

## 2. Thông tin GPU & PyTorch CUDA
- **PyTorch Version**: `{pt.get('torch_version', 'N/A')}`
- **CUDA Compiled Version**: `{pt.get('cuda_version', 'N/A')}`
- **Hỗ trợ bfloat16**: `{pt.get('is_bf16_supported', False)}`
- **Danh sách GPU**:
"""
    for gpu in pt.get("gpus", []):
        md_content += f"  - **GPU {gpu['id']}**: `{gpu['name']}` | VRAM: `{gpu['vram_gb']} GB` | Compute Capability: `{gpu['compute_capability']}`\n"

    md_content += "\n## 3. Danh sách Phiên bản Thư viện AI\n\n| Thư viện | Phiên bản |\n| :--- | :--- |\n"
    for pkg, ver in report["packages"].items():
        md_content += f"| `{pkg}` | `{ver}` |\n"

    with open("ENVIRONMENT_INFO.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    with open("sys_info.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n✅ Đã lưu kết quả báo cáo ra 2 tệp:")
    print("   1. ENVIRONMENT_INFO.md (Tài liệu Markdown dạng bảng)")
    print("   2. sys_info.json (Dữ liệu cấu hình định dạng JSON)")

if __name__ == "__main__":
    audit_environment()
