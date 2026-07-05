"""Metrics for phoneme2token."""
from typing import Dict, List

import torch


METRIC_NAMES: List[str] = [
    "token_acc",
    "nonpad_acc",
    "pad_acc",
    "pad_ratio",
    "loss_all",
    "frontend_blank_ratio",
    "frontend_nonblank_frames",
    "frontend_nonblank_frames_restored",
    "frontend_phone_frames_100hz",
    "frontend_identity_pass_rate",
]


@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_pad_id: int,
    ignore_index: int = -100,
) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    valid = labels.ne(ignore_index)
    if not valid.any():
        return {name: 0.0 for name in METRIC_NAMES}

    correct = pred.eq(labels) & valid
    pad_mask = labels.eq(token_pad_id) & valid
    nonpad_mask = labels.ne(token_pad_id) & valid

    metrics = {
        "token_acc": correct.sum().float().div(valid.sum()).item(),
        "pad_ratio": pad_mask.sum().float().div(valid.sum()).item(),
        "nonpad_acc": 0.0,
        "pad_acc": 0.0,
    }
    if nonpad_mask.any():
        metrics["nonpad_acc"] = (correct & nonpad_mask).sum().float().div(nonpad_mask.sum()).item()
    if pad_mask.any():
        metrics["pad_acc"] = (correct & pad_mask).sum().float().div(pad_mask.sum()).item()
    return metrics
