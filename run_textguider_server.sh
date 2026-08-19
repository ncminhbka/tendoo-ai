#!/usr/bin/env bash
# ==============================================================================
# Script chạy thử nghiệm FLUX.2 Klein 4B Base + TextGuider trên GPU Server (Viettel)
# Paper: TextGuider (arXiv:2512.09350)
# ==============================================================================

set -e

# Đảm bảo đứng từ thư mục gốc của repository
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "🚀 Đang khởi động thử nghiệm FLUX.2 Klein 4B Base + TextGuider..."
echo "📂 Thư mục làm việc: $(pwd)"
echo "=================================================================="

# 1. Kích hoạt môi trường conda 'tendoo_ai' nếu chưa được kích hoạt
if [ "$CONDA_DEFAULT_ENV" != "tendoo_ai" ]; then
    echo "🔄 Đang kích hoạt môi trường conda 'tendoo_ai'..."
    # Thử hook conda cho shell bash
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null || true)"
        conda activate tendoo_ai 2>/dev/null || true
    fi
    # Nếu conda nằm trong các đường dẫn tiêu chuẩn trên server
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

if [ "$CONDA_DEFAULT_ENV" == "tendoo_ai" ]; then
    echo "✅ Đang sử dụng môi trường: $CONDA_DEFAULT_ENV (Python: $(python --version 2>&1))"
else
    echo "⚠️  Cảnh báo: Chưa thể tự động kích hoạt 'tendoo_ai'. Hãy chắc chắn bạn đã gõ 'conda activate tendoo_ai' trước khi chạy script."
fi

# 2. Kiểm tra GPU
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️  Không tìm thấy lệnh nvidia-smi."
fi

# 3. Tự động xác định đường dẫn Model FLUX.2 Klein Base
# Ưu tiên tham số truyền vào $1 -> đường dẫn cố định trong persistent-data -> HuggingFace ID
MODEL_PATH=""
if [ -n "$1" ]; then
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

SEED=${2:-42}
STEPS=${3:-50}
GUIDANCE=${4:-4.0}
ALPHA=${5:-60.0}

OUTPUT_DIR="outputs/textguider_server_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "------------------------------------------------------------------"
echo "🔹 Model Path / ID:  $MODEL_PATH"
echo "🔹 Seed:             $SEED"
echo "🔹 Steps:            $STEPS"
echo "🔹 Guidance Scale:   $GUIDANCE"
echo "🔹 TextGuider α:     $ALPHA"
echo "🔹 Output Dir:       $OUTPUT_DIR"
echo "------------------------------------------------------------------"

# Thiết lập PYTHONPATH
export PYTHONPATH=".:$PYTHONPATH"

# Chạy script Python tạo ảnh và so sánh Base vs TextGuider
python generate_textguider_samples.py \
    --model-id "$MODEL_PATH" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --guidance-scale "$GUIDANCE" \
    --alpha "$ALPHA" \
    --t-guide-ratio 0.25 \
    --amo-c 0.5 \
    --compare \
    --output-dir "$OUTPUT_DIR"

echo "=================================================================="
echo "🎉 Hoàn tất sinh ảnh! Kết quả được lưu tại: $OUTPUT_DIR"
echo "=================================================================="
