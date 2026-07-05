"""
Wav2Phoneme Metrics Module

计算模型损失和额外指标
"""
import torch
import numpy as np
from typing import Dict, Optional

from .metric import compute_metrics_batch


# 需要记录的指标名称
METRIC_NAMES = ["frame_diff", "rel_frame_diff", "frame_per", "seq_per"]


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    mode: str = "forced_alignment",
    input_lengths: Optional[torch.Tensor] = None,
    label_lengths: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    计算 Wav2Phoneme 的额外指标

    Args:
        logits: [B, T', num_phonemes] - 模型输出
        labels: [B, T'] - 标签
        label_mask: [B, T'] - 标签掩码
        attention_mask: [B, T] - 音频注意力掩码
        mode: 训练模式
        input_lengths: [B] - encoder 输出长度（CTC 模式）
        label_lengths: [B] - 标签长度（CTC 模式）

    Returns:
        dict containing:
            - frame_diff: 帧数差异绝对值
            - rel_frame_diff: 相对帧数差异
            - frame_per: 帧级别 PER
            - seq_per: 序列级别 PER
    """
    frame_diffs = []
    rel_frame_diffs = []

    # 计算帧数差异
    batch_size = logits.shape[0]
    for i in range(batch_size):
        # 音频帧数 (wav2vec 下采样 320 倍)
        if input_lengths is not None:
            audio_len = int(input_lengths[i].item())
        elif attention_mask is not None:
            audio_len = attention_mask[i].sum().item() // 320
        else:
            audio_len = logits.shape[1]

        # 标签帧数
        if label_lengths is not None:
            label_len = int(label_lengths[i].item())
        elif label_mask is not None:
            label_len = label_mask[i].sum().item()
        else:
            label_len = (labels[i] != -1).sum().item()

        if label_len > 0:
            abs_diff = abs(audio_len - label_len)
            rel_diff = abs_diff / label_len
            frame_diffs.append(abs_diff)
            rel_frame_diffs.append(rel_diff)

    # CTC labels live on a different time axis than encoder-frame logits, so
    # the FA/interpolate frame-PER helpers do not apply directly.
    if mode == "ctc":
        return {
            "frame_diff": float(np.mean(frame_diffs)) if frame_diffs else 0.0,
            "rel_frame_diff": float(np.mean(rel_frame_diffs)) if rel_frame_diffs else 0.0,
            "frame_per": 0.0,
            "seq_per": 0.0,
        }

    # 计算 PER metrics
    with torch.no_grad():
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        label_mask_np = label_mask.detach().cpu().numpy() if label_mask is not None else None

        metrics = compute_metrics_batch(logits_np, labels_np, label_mask_np)

    return {
        "frame_diff": float(np.mean(frame_diffs)) if frame_diffs else 0.0,
        "rel_frame_diff": float(np.mean(rel_frame_diffs)) if rel_frame_diffs else 0.0,
        "frame_per": metrics["fper"],
        "seq_per": metrics["sper"],
    }
