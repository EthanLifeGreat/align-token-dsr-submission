"""
Phoneme2Mel Metrics Module

计算模型损失和额外指标
"""
import torch
from typing import Dict, Optional

from .loss import phoneme2mel_loss


# 需要记录的指标名称
METRIC_NAMES = ["loss_after_postnet", "loss_before_postnet"]


def compute_metrics(
    mel_pred: torch.Tensor,
    mel_pred_before: torch.Tensor,
    mel_target: torch.Tensor,
    mel_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    计算 Phoneme2Mel 的损失和指标

    Args:
        mel_pred: [B, T_pred, mel_dim] - postnet后的预测
        mel_pred_before: [B, T_pred, mel_dim] - postnet前的预测
        mel_target: [B, T_mel, mel_dim] - 目标mel频谱
        mel_mask: [B, T_mel] - mel mask (可选)

    Returns:
        dict containing:
            - loss: 总损失 (用于反向传播)
            - loss_after_postnet: postnet后的L1损失
            - loss_before_postnet: postnet前的L1损失
    """
    loss_dict = phoneme2mel_loss(
        mel_pred=mel_pred,
        mel_pred_before=mel_pred_before,
        mel_target=mel_target,
        loss_mask=mel_mask,
        return_dict=True,
    )

    return loss_dict
