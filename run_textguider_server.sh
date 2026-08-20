#!/usr/bin/env bash
# ==============================================================================
# Script chạy thử nghiệm FLUX.2 Klein 4B Base + TextGuider trên GPU Server (Viettel)
# Paper: TextGuider (arXiv:2512.09350)
#
# Cập nhật: xem ARCHITECTURE_NOTES.md — Base model (không distill) dùng CFG
# thật (guidance mặc định 4.0 đã đúng từ trước). Mặc định script này chạy
# strict_mode=True (raise ngay nếu hook/token-alignment thất bại); đặt biến
# môi trường NO_STRICT=1 để tắt nếu bạn đã xác nhận pipeline chạy đúng.
# ==============================================================================

set -e

# Đảm bảo đứng từ thư mục gốc của repository
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "🚀 Đang khởi động TextGuider cho FLUX.2 Klein 4B Base..."
echo "📂 Thư mục làm việc: $(pwd)"
echo "=================================================================="

# 1. Kích hoạt môi trường ảo (.venv / uv hoặc conda)
if [ -f ".venv/bin/activate" ]; then
    echo "⚡ Kích hoạt môi trường .venv trong thư mục hiện tại..."
    source .venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    echo "⚡ Kích hoạt môi trường .venv ở thư mục cha..."
    source ../.venv/bin/activate
elif [ -f "../../.venv/bin/activate" ]; then
    echo "⚡ Kích hoạt môi trường .venv ở thư mục gốc..."
    source ../../.venv/bin/activate
elif [ "$CONDA_DEFAULT_ENV" != "tendoo_ai" ]; then
    echo "🔄 Đang tìm môi trường conda 'tendoo_ai'..."
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null || true)"
        conda activate tendoo_ai 2>/dev/null || true
    fi
    if [ "$CONDA_DEFAULT_ENV" != "tendoo_ai" ]; then
        for CONDA_PATH in "/opt/conda/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
            if [ -f "$CONDA_PATH" ]; then
                eval "$($CONDA_PATH shell.bash hook 2>/dev/null || true)"
                conda activate tendoo_ai 2>/dev/null || true
                break
            fi
        done
    fi
fi

echo "✅ Python runtime: $(which python 2>/dev/null || echo 'not found') ($(python --version 2>&1 || true))"

# 2. Kiểm tra GPU
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️  Không tìm thấy lệnh nvidia-smi."
fi

# 3. Tự động xác định đường dẫn Model FLUX.2 Klein Base trong persistent-data
MODEL_PATH=""
if [ -n "$1" ] && [ "$1" != "--compare" ]; then
    MODEL_PATH="$1"
elif [ -d "/persistent-data/FLUX.2-klein-base-4B" ]; then
    MODEL_PATH="/persistent-data/FLUX.2-klein-base-4B"
elif [ -d "../persistent-data/FLUX.2-klein-base-4B" ]; then
    MODEL_PATH="../persistent-data/FLUX.2-klein-base-4B"
elif [ -d "../../persistent-data/FLUX.2-klein-base-4B" ]; then
    MODEL_PATH="../../persistent-data/FLUX.2-klein-base-4B"
elif [ -d "persistent-data/FLUX.2-klein-base-4B" ]; then
    MODEL_PATH="persistent-data/FLUX.2-klein-base-4B"
elif [ -d "$HOME/persistent-data/FLUX.2-klein-base-4B" ]; then
    MODEL_PATH="$HOME/persistent-data/FLUX.2-klein-base-4B"
else
    MODEL_PATH="black-forest-labs/FLUX.2-klein-base-4B"
fi

SEED=42
STEPS=50
GUIDANCE=4.0
ALPHA=60.0
COMPARE_FLAG=""
STRICT_FLAG=""

# Kiểm tra nếu người dùng muốn so sánh (--compare)
for arg in "$@"; do
    if [ "$arg" == "--compare" ]; then
        COMPARE_FLAG="--compare"
    fi
done

# Đặt NO_STRICT=1 trước khi chạy script này nếu muốn tắt strict_mode
if [ "$NO_STRICT" == "1" ]; then
    STRICT_FLAG="--no-strict"
    echo "⚠️  NO_STRICT=1: chạy với strict_mode=False (fallback êm ái khi lỗi)."
fi

OUTPUT_DIR="outputs/textguider_server_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "------------------------------------------------------------------"
echo "🔹 Model Path / ID:  $MODEL_PATH"
echo "🔹 Seed:             $SEED"
echo "🔹 Steps:            $STEPS (Guidance steps: 12, guidance CFG thật cho model Base)"
echo "🔹 Guidance Scale:   $GUIDANCE"
echo "🔹 TextGuider α:     $ALPHA"
echo "🔹 Strict mode:      $([ -z "$STRICT_FLAG" ] && echo "on (mặc định)" || echo "off")"
echo "🔹 Mode:             Chạy trực tiếp TextGuider ${COMPARE_FLAG:+(kèm so sánh Base)}"
echo "🔹 Output Dir:       $OUTPUT_DIR"
echo "------------------------------------------------------------------"

# Thiết lập PYTHONPATH
export PYTHONPATH=".:$PYTHONPATH"

# Chạy script Python tạo ảnh
python generate_textguider_samples.py \
    --model-id "$MODEL_PATH" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --guidance-scale "$GUIDANCE" \
    --alpha "$ALPHA" \
    --t-guide-ratio 0.25 \
    --amo-c 0.5 \
    $COMPARE_FLAG \
    $STRICT_FLAG \
    --output-dir "$OUTPUT_DIR"

echo "=================================================================="
echo "🎉 Hoàn tất sinh ảnh TextGuider! Kết quả lưu tại: $OUTPUT_DIR"
echo "=================================================================="
