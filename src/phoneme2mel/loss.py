"""
损失函数定义
使用 Tacotron 风格的双L1 Loss，并在loss计算时使用插值对齐phoneme和mel帧
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def phoneme2mel_loss(
    mel_pred: torch.Tensor,
    mel_pred_before: torch.Tensor,
    mel_target: torch.Tensor,
    loss_mask: torch.Tensor = None,
    return_dict: bool = False,
) -> torch.Tensor:
    """
    Phoneme2Mel 损失函数 (Tacotron风格)

    使用两个L1 Loss:
    1. postnet之前的预测 vs mel_target
    2. postnet之后的预测 vs mel_target

    如果mel_pred和mel_target长度不匹配，将mel_target插值到mel_pred长度

    Args:
        mel_pred: [B, T_pred, mel_dim] - postnet后的预测
        mel_pred_before: [B, T_pred, mel_dim] - postnet前的预测
        mel_target: [B, T_mel, mel_dim] - 目标mel频谱
        loss_mask: [B, T_mel] - loss mask (可选)
        return_dict: bool - 是否返回包含各个loss的字典

    Returns:
        loss: scalar 或 dict (当return_dict=True时)
    """
    T_pred = mel_pred.size(1)
    T_mel = mel_target.size(1)

    # 如果长度不匹配，插值mel_target到mel_pred长度
    if T_pred != T_mel:
        # mel_target: [B, T_mel, mel_dim] -> [B, mel_dim, T_mel]
        mel_target_t = mel_target.transpose(1, 2)

        # 插值到 T_pred
        mel_target_interp = F.interpolate(mel_target_t, size=T_pred, mode='linear', align_corners=False)

        # 转回来: [B, T_pred, mel_dim]
        mel_target_interp = mel_target_interp.transpose(1, 2)

        # 同时插值loss_mask
        if loss_mask is not None:
            # loss_mask: [B, T_mel] -> [B, 1, T_mel]
            loss_mask = F.interpolate(loss_mask.unsqueeze(1).float(), size=T_pred, mode='linear', align_corners=False).squeeze(1)
    else:
        mel_target_interp = mel_target

    # 计算 L1 Loss (使用reduction="none"以便mask)
    l1_criterion = nn.L1Loss(reduction="none")

    loss_after = l1_criterion(mel_pred, mel_target_interp)
    loss_before = l1_criterion(mel_pred_before, mel_target_interp)

    # 求所有维度的mean (除了batch)
    loss_after = loss_after.mean(dim=-1)  # [B, T]
    loss_before = loss_before.mean(dim=-1)  # [B, T]

    # 计算总loss
    loss = loss_after + loss_before

    # 应用mask
    if loss_mask is not None:
        # loss_mask: [B, T_pred]
        loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        loss_after = (loss_after * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        loss_before = (loss_before * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    else:
        loss = loss.mean()
        loss_after = loss_after.mean()
        loss_before = loss_before.mean()

    if return_dict:
        return {
            "loss": loss,
            "loss_after_postnet": loss_after,
            "loss_before_postnet": loss_before,
        }
    return loss


def get_loss_fn(mode: str = "phoneme2mel", **kwargs):
    """
    获取损失函数

    Args:
        mode: "phoneme2mel"
        **kwargs: 额外参数

    Returns:
        损失函数
    """
    if mode == "phoneme2mel":
        return phoneme2mel_loss
    else:
        raise ValueError(f"Unknown mode: {mode}")
