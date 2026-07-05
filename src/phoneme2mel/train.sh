#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

STAGE="${1:-pretrain}"
GPUS="${2:-0}"
FINETUNE_CKPT_OVERRIDE="${3:-}"
CONFIG_ROOT="${CONFIG_ROOT:-src/phoneme2mel/configs/rope192}"
PRETRAIN_EXP_NAME="${PRETRAIN_EXP_NAME:-rope192/AISHELL-2-95_5-pretrain}"
FINETUNE_EXP_NAME="${FINETUNE_EXP_NAME:-rope192/CSMSC-finetune}"

launch_single_stage() {
    local stage="$1"
    local gpus="$2"
    local finetune_ckpt_override="${3:-}"
    local config_name=""

    case "$stage" in
        pretrain)
            config_name="AISHELL-2-pretrain"
            ;;
        finetune)
            config_name="CSMSC-finetune"
            ;;
        *)
            echo "Unsupported stage: $stage"
            return 1
            ;;
    esac

    local train_config_path="$CONFIG_ROOT/$config_name/train_config.yaml"
    local tmp_train_config=""
    if [[ "$stage" == "finetune" && -n "$finetune_ckpt_override" ]]; then
        # The unified trainer is launched in the background. Keep the override
        # file under the experiment tree so it still exists when Python opens it.
        local override_dir="$PROJECT_ROOT/results/phoneme2mel/$FINETUNE_EXP_NAME"
        mkdir -p "$override_dir"
        tmp_train_config="$override_dir/train_config.override.yaml"
        sed "s|^finetune_from_checkpoint:.*$|finetune_from_checkpoint: ${finetune_ckpt_override}|" \
            "$train_config_path" > "$tmp_train_config"
        train_config_path="$tmp_train_config"
    fi

    "$PROJECT_ROOT/src/train_utils/train.sh" \
        --model phoneme2mel \
        --model-config "$CONFIG_ROOT/$config_name/config.yaml" \
        --train-config "$train_config_path" \
        --gpus "$gpus"

}

wait_for_stage_exit() {
    local output_dir="$1"
    local stage_name="$2"
    local pid_file="$output_dir/train.pid"

    while [[ ! -f "$pid_file" ]]; do
        sleep 2
    done

    local pid
    pid="$(cat "$pid_file")"
    if [[ -z "$pid" ]]; then
        echo "Empty PID file for $stage_name: $pid_file"
        return 1
    fi

    echo "Waiting for $stage_name to finish (pid=$pid)..."
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
    done
}

case "$STAGE" in
    pretrain)
        CONFIG_NAME="AISHELL-2-pretrain"
        ;;
    finetune)
        CONFIG_NAME="CSMSC-finetune"
        ;;
    all)
        CONFIG_NAME=""
        ;;
    *)
        echo "Usage: bash src/phoneme2mel/train.sh [pretrain|finetune|all] [gpus] [finetune_ckpt_override]"
        exit 1
        ;;
esac

if [[ "$STAGE" == "all" ]]; then
    PRETRAIN_OUTPUT_DIR="$PROJECT_ROOT/results/phoneme2mel/$PRETRAIN_EXP_NAME"
    PRETRAIN_FINAL_DIR="$PRETRAIN_OUTPUT_DIR/final"

    launch_single_stage pretrain "$GPUS"
    wait_for_stage_exit "$PRETRAIN_OUTPUT_DIR" "pretrain"

    if [[ ! -d "$PRETRAIN_FINAL_DIR" ]]; then
        echo "Pretrain finished but final checkpoint directory not found: $PRETRAIN_FINAL_DIR"
        exit 1
    fi

    launch_single_stage finetune "$GPUS" "$PRETRAIN_FINAL_DIR"
    exit 0
fi

launch_single_stage "$STAGE" "$GPUS" "$FINETUNE_CKPT_OVERRIDE"
