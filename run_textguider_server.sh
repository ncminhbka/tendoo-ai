#!/usr/bin/env bash
# ==============================================================================
# Script chạy thử nghiệm FLUX.2 Klein 4B Base + TextGuider trên GPU Server
# Paper: TextGuider (arXiv:2512.09350)
# ==============================================================================

set -e

echo "=================================================================="
echo "🚀 Đang khởi động thử nghiệm FLUX.2 Klein 4B Base + TextGuider..."
echo "=================================================================="

# Kiểm tra môi trường GPU
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "⚠️  Không tìm thấy lệnh nvidia-smi. Đang chạy ở chế độ CPU/Dry-run."
fi

# Thiết lập thư mục output
OUTPUT_DIR="outputs/textguider_server_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

MODEL_ID=${1:-"black-forest-labs/FLUX.2-klein-base-4B"}
SEED=${2:-42}
STEPS=${3:-50}
GUIDANCE=${4:-4.0}
ALPHA=${5:-60.0}

echo "🔹 Model ID:        $MODEL_ID"
echo "🔹 Seed:            $SEED"
echo "🔹 Steps:           $STEPS"
echo "🔹 Guidance Scale:  $GUIDANCE"
echo "🔹 TextGuider α:    $ALPHA"
echo "🔹 Output Dir:      $OUTPUT_DIR"
echo "------------------------------------------------------------------"

# Chạy script Python tạo ảnh so sánh (Base vs TextGuider)
python generate_textguider_samples.py \
    --model-id "$MODEL_ID" \
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
