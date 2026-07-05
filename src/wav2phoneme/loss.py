"""
损失函数定义
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def forced_alignment_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Forced Alignment 损失函数 (CrossEntropy)

    注意: 插值已在 Collator 中完成，此处假设 labels 与 logits 长度已匹配

    Args:
        logits: [B, T, num_phonemes]
        labels: [B, T]
        label_mask: [B, T], 可选，用于加权
        ignore_index: 忽略的标签索引

    Returns:
        loss: scalar
    """
    # 转换维度: [B, T, C] -> [B, C, T]
    logits = logits.transpose(1, 2)

    # 计算 cross entropy loss
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")

    # 应用 label_mask 加权
    if label_mask is not None:
        loss = loss * label_mask.float()
        loss = loss.sum() / label_mask.sum().clamp(min=1)
    else:
        loss = loss.mean()

    return loss


def ctc_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
    blank: int = 0,
    zero_infinity: bool = True,
) -> torch.Tensor:
    """
    CTC 损失函数
    
    Args:
        logits: [B, T, num_phonemes]
        labels: [B, T_label]
        input_lengths: [B], encoder 输出帧数
        label_lengths: [B]
        blank: blank token ID
        zero_infinity: 是否将无穷大损失置零
    
    Returns:
        loss: scalar
    """
    # PyTorch CTC expects log-probabilities in [T, B, C].
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

    loss = F.ctc_loss(
        log_probs,
        labels,
        input_lengths=input_lengths,
        target_lengths=label_lengths,
        blank=blank,
        zero_infinity=zero_infinity,
        reduction="mean",
    )
    
    return loss


def interpolate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Interpolate 损失函数 (同 forced_alignment_loss)
    
    Args:
        logits: [B, T, num_phonemes]
        labels: [B, T]
        label_mask: [B, T], 可选
        ignore_index: 忽略的标签索引
    
    Returns:
        loss: scalar
    """
    return forced_alignment_loss(logits, labels, label_mask, ignore_index)


def get_loss_fn(mode: str, **kwargs):
    """
    根据模式获取损失函数
    
    Args:
        mode: "forced_alignment", "ctc", "interpolate"
        **kwargs: 额外参数
    
    Returns:
        损失函数
    """
    mode = mode.lower()
    
    if mode == "forced_alignment":
        return lambda logits, labels, **kw: forced_alignment_loss(
            logits, labels, 
            label_mask=kw.get("label_mask"),
            ignore_index=kwargs.get("ignore_index", -1),
        )
    elif mode == "ctc":
        return lambda logits, labels, **kw: ctc_loss(
            logits, labels,
            input_lengths=kw["input_lengths"],
            label_lengths=kw["label_lengths"],
            blank=kwargs.get("blank", 0),
            zero_infinity=kwargs.get("zero_infinity", True),
        )
    elif mode == "interpolate":
        return lambda logits, labels, **kw: interpolate_loss(
            logits, labels,
            label_mask=kw.get("label_mask"),
            ignore_index=kwargs.get("ignore_index", -1),
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
