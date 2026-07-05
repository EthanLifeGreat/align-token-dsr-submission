#!/bin/bash

# ============================================
# Wav2Phoneme Training Script
# ============================================
# 调用通用训练脚本，硬编码配置路径和 GPU

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ============================================
# 配置（根据需要修改）
# ============================================
CONFIG_NAME="CDSD-FA"
GPUS="1,2,3,5"

# ============================================
# 调用通用训练脚本
# ============================================
"$PROJECT_ROOT/src/train_utils/train.sh" \
    --model wav2phoneme \
    --model-config "src/wav2phoneme/configs/$CONFIG_NAME/config.yaml" \
    --train-config "src/wav2phoneme/configs/$CONFIG_NAME/train_config.yaml" \
    --gpus "$GPUS"
