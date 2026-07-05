#!/bin/bash

# ============================================
# Unified Training Script
# ============================================
# Usage:
#   ./src/train.sh --model <model_name> --model-config <path> --train-config <path> [--gpus <gpu_ids>] [--conda-env <env_name>]
#
# Example:
#   ./src/train.sh --model phoneme2mel \
#       --model-config src/phoneme2mel/configs/CSMSC-finetune/config.yaml \
#       --train-config src/phoneme2mel/configs/CSMSC-finetune/train_config.yaml \
#       --gpus "2,3"
#
#   ./src/train.sh --model wav2phoneme \
#       --model-config src/wav2phoneme/configs/AISHELL2-FA/config.yaml \
#       --train-config src/wav2phoneme/configs/AISHELL2-FA/train_config.yaml \
#       --gpus "4,5"

set -e

# Some environments (including certain CI/tool runners) aggressively reap child
# processes when the parent shell exits. For debugging, allow running the trainer
# in the foreground so we can observe whether eval hangs (e.g. NCCL deadlock).
# Usage:
#   TRAIN_FOREGROUND=1 bash src/<model>/train.sh
TRAIN_FOREGROUND="${TRAIN_FOREGROUND:-0}"

# ============================================
# 解析参数
# ============================================
MODEL=""
MODEL_CONFIG=""
TRAIN_CONFIG=""
GPUS=""
CONDA_ENV="dsr"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model-config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --train-config)
            TRAIN_CONFIG="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 验证必需参数
if [ -z "$MODEL" ]; then
    echo "Error: --model is required"
    exit 1
fi
if [ -z "$MODEL_CONFIG" ]; then
    echo "Error: --model-config is required"
    exit 1
fi
if [ -z "$TRAIN_CONFIG" ]; then
    echo "Error: --train-config is required"
    exit 1
fi

# ============================================
# GPU 配置
# ============================================
if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
else
    # 默认使用 GPU 0
    export CUDA_VISIBLE_DEVICES="0"
fi

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============================================
# 从 train_config.yaml 读取配置
# ============================================
EXP_NAME=$(grep -E "^exp_name:" "$TRAIN_CONFIG" | awk '{print $2}' | tr -d '"')

# 兼容 batch_size 和 effective_batch_size 两种配置
EFFECTIVE_BS=$(grep -E "^effective_batch_size:" "$TRAIN_CONFIG" | awk '{print $2}')
if [ -z "$EFFECTIVE_BS" ]; then
    EFFECTIVE_BS=$(grep -E "^batch_size:" "$TRAIN_CONFIG" | awk '{print $2}')
fi

BATCH_SIZE_UNIT=$(grep -E "^batch_size_unit:" "$TRAIN_CONFIG" | awk '{print $2}' | tr -d '"')
if [ -z "$BATCH_SIZE_UNIT" ]; then
    BATCH_SIZE_UNIT="sample"
fi

ACC_STEPS=$(grep -E "^accumulation_steps:" "$TRAIN_CONFIG" | awk '{print $2}')

# 默认值
if [ -z "$ACC_STEPS" ]; then
    ACC_STEPS=1
fi

OUTPUT_DIR="$PROJECT_ROOT/results/$MODEL/$EXP_NAME"
# 生成基于时间的唯一日志文件名 (格式: train_YYYYMMDD_HHMMSS.log)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_${TIMESTAMP}.log"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 计算 GPU 数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | grep -c '[0-9]')

# 计算 per-device batch 控制值
# sample 模式: effective_batch_size = per_device_bs * num_gpus * accumulation_steps
# token 模式: effective_batch_size = per_device_token_budget * num_gpus * accumulation_steps
PER_DEVICE_BS=$((EFFECTIVE_BS / (NUM_GPUS * ACC_STEPS)))

if [ "$PER_DEVICE_BS" -le 0 ]; then
    echo "Error: effective_batch_size ($EFFECTIVE_BS) is too small for num_gpus ($NUM_GPUS) * accumulation_steps ($ACC_STEPS)"
    exit 1
fi

# 验证是否整除
if [ -n "$EFFECTIVE_BS" ] && [ $((PER_DEVICE_BS * NUM_GPUS * ACC_STEPS)) -ne $EFFECTIVE_BS ]; then
    echo "[Warning] effective_batch_size ($EFFECTIVE_BS) cannot be evenly divided by num_gpus ($NUM_GPUS) * accumulation_steps ($ACC_STEPS)"
    if [ "$BATCH_SIZE_UNIT" = "token" ]; then
        echo "[Warning] Per device token budget will be $PER_DEVICE_BS (effective = $((PER_DEVICE_BS * NUM_GPUS * ACC_STEPS)))"
    else
        echo "[Warning] Per device batch size will be $PER_DEVICE_BS (effective = $((PER_DEVICE_BS * NUM_GPUS * ACC_STEPS)))"
    fi
fi

# ============================================
# PYTHONPATH 设置
# ============================================
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Ensure python logs flush to train.log promptly (torchrun + redirection otherwise buffers).
export PYTHONUNBUFFERED=1

# ============================================
# CUDA 环境配置
# ============================================
export CUDA_LAUNCH_BLOCKING=0   # 1 for debug

# MASTER_PORT 可从外部传入，否则根据当前分钟数动态分配 (29600-29659)
if [ -z "$MASTER_PORT" ]; then
    CURRENT_MINUTE=$(date +%M)
    # 移除前导零，避免八进制解析问题
    CURRENT_MINUTE=$((10#$CURRENT_MINUTE))
    export MASTER_PORT=$((29600 + CURRENT_MINUTE))
fi

# ============================================
# Conda 环境激活
# ============================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# ============================================
# 打印配置信息
# ============================================
echo "============================================"
echo "Training: $MODEL"
echo "============================================"
echo "Project Root:       $PROJECT_ROOT"
echo "Model Config:       $MODEL_CONFIG"
echo "Train Config:       $TRAIN_CONFIG"
echo "Output Dir:         $OUTPUT_DIR"
echo "Log File:           $LOG_FILE"
echo "============================================"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Conda Env:            $CONDA_ENV"
echo "Num GPUs:             $NUM_GPUS"
echo "Batch Size Unit:      $BATCH_SIZE_UNIT"
echo "Effective Batch Size: $EFFECTIVE_BS"
echo "Accumulation Steps:   $ACC_STEPS"
if [ "$BATCH_SIZE_UNIT" = "token" ]; then
    echo "Per Device Token Cap: $PER_DEVICE_BS"
else
    echo "Per Device BS:        $PER_DEVICE_BS"
fi
echo "============================================"

# ============================================
# 启动训练
# ============================================
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "[DDP Mode] Using torchrun with $NUM_GPUS processes"
    if [ "$TRAIN_FOREGROUND" = "1" ]; then
        # Still write a log file, but keep stdout visible for debugging.
        torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
            src/train_utils/train.py \
            --model "$MODEL" \
            --model-config "$MODEL_CONFIG" \
            --train-config "$TRAIN_CONFIG" \
            --output-dir "$OUTPUT_DIR" \
            --per-device-batch-size "$PER_DEVICE_BS" \
            2>&1 | tee "$LOG_FILE"
        exit ${PIPESTATUS[0]}
    fi

    nohup torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        src/train_utils/train.py \
        --model "$MODEL" \
        --model-config "$MODEL_CONFIG" \
        --train-config "$TRAIN_CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        --per-device-batch-size "$PER_DEVICE_BS" \
        > "$LOG_FILE" 2>&1 &
else
    echo "[Single GPU Mode] Using python"
    if [ "$TRAIN_FOREGROUND" = "1" ]; then
        python -u src/train_utils/train.py \
            --model "$MODEL" \
            --model-config "$MODEL_CONFIG" \
            --train-config "$TRAIN_CONFIG" \
            --output-dir "$OUTPUT_DIR" \
            --per-device-batch-size "$PER_DEVICE_BS" \
            2>&1 | tee "$LOG_FILE"
        exit ${PIPESTATUS[0]}
    fi

    nohup python -u src/train_utils/train.py \
        --model "$MODEL" \
        --model-config "$MODEL_CONFIG" \
        --train-config "$TRAIN_CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        --per-device-batch-size "$PER_DEVICE_BS" \
        > "$LOG_FILE" 2>&1 &
fi

PID=$!
echo $PID > "$OUTPUT_DIR/train.pid"

# ============================================
# 打印监控命令
# ============================================
echo ""
echo "Training started in background!"
echo ""
echo "============================================"
echo "Monitor Commands:"
echo "============================================"
echo ""
echo "# View log (tail -f):"
echo "    tail -f $LOG_FILE"
echo ""
echo "# View log (cat):"
echo "    cat $LOG_FILE"
echo ""
echo "# TensorBoard:"
echo "    tensorboard --logdir=$OUTPUT_DIR/tensorboard --port=6006"
echo ""
echo "# Find process:"
echo "    pgrep -f \"$OUTPUT_DIR\""
echo ""
echo "# Kill process:"
echo "    pkill -f \"$OUTPUT_DIR\""
echo "    # or"
echo "    kill $PID"
echo ""
echo "============================================"
