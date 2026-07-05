"""
评估指标 - PER (Phoneme Error Rate)
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from jiwer import wer


def compute_frame_per(
    pred_ids: np.ndarray,
    label_ids: np.ndarray,
    ignore_index: int = -1,
) -> float:
    """
    计算帧级别 Phoneme Error Rate
    
    Args:
        pred_ids: 预测的音素 ID 序列
        label_ids: 真实的音素 ID 序列
        ignore_index: 忽略的标签索引
    
    Returns:
        frame PER
    """
    # 过滤 ignore_index
    valid_mask = label_ids != ignore_index
    valid_pred = pred_ids[valid_mask]
    valid_label = label_ids[valid_mask]
    
    if len(valid_label) == 0:
        return 0.0
    
    # 计算错误率
    errors = (valid_pred != valid_label).sum()
    total = len(valid_label)
    
    return errors / total


def compute_sequence_per(
    pred_ids: np.ndarray,
    label_ids: np.ndarray,
) -> float:
    """
    计算序列级别 PER（去重后比较）
    
    Args:
        pred_ids: 预测的音素 ID 序列
        label_ids: 真实的音素 ID 序列
    
    Returns:
        sequence PER
    """
    # 去重
    pred_dedup = deduplicate(pred_ids)
    label_dedup = deduplicate(label_ids)
    
    # 转为字符串计算 WER
    pred_str = " ".join(map(str, pred_dedup))
    label_str = " ".join(map(str, label_dedup))
    
    if len(label_str) == 0:
        return 0.0
    
    return wer(label_str, pred_str)


def compute_smoothed_per(
    pred_ids: np.ndarray,
    label_ids: np.ndarray,
    ignore_index: int = -1,
) -> float:
    """
    计算平滑后的序列 PER
    使用编辑距离归一化
    
    Args:
        pred_ids: 预测的音素 ID 序列
        label_ids: 真实的音素 ID 序列
        ignore_index: 忽略的标签索引
    
    Returns:
        smoothed PER
    """
    # 过滤
    valid_mask = label_ids != ignore_index
    pred_filtered = pred_ids[valid_mask][:len(label_ids[valid_mask])]
    label_filtered = label_ids[valid_mask]
    
    if len(label_filtered) == 0:
        return 0.0
    
    # 去重
    pred_dedup = deduplicate(pred_filtered)
    label_dedup = deduplicate(label_filtered)
    
    # 计算编辑距离
    edit_dist = levenshtein_distance(pred_dedup.tolist(), label_dedup.tolist())
    
    return edit_dist / max(len(label_dedup), 1)


def deduplicate(arr: np.ndarray) -> np.ndarray:
    """
    去重序列中连续相同的元素
    
    Args:
        arr: 输入序列
    
    Returns:
        去重后的序列
    """
    if len(arr) == 0:
        return arr
    
    result = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:
            result.append(arr[i])
    
    return np.array(result)


def levenshtein_distance(s1: List, s2: List) -> int:
    """
    计算两个序列之间的编辑距离
    
    Args:
        s1: 序列1
        s2: 序列2
    
    Returns:
        编辑距离
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def compute_metrics_batch(
    logits: np.ndarray,
    labels: np.ndarray,
    label_mask: Optional[np.ndarray] = None,
    ignore_index: int = -1,
) -> Dict[str, float]:
    """
    批量计算评估指标
    
    Args:
        logits: [B, T, C]
        labels: [B, T]
        label_mask: [B, T]
        ignore_index: 忽略的标签索引
    
    Returns:
        {"fper": float, "sper": float, "sper_s": float}
    """
    batch_size = logits.shape[0]
    pred_ids = logits.argmax(axis=-1)
    
    fper_list = []
    sper_list = []
    sper_s_list = []
    
    for i in range(batch_size):
        pred = pred_ids[i]
        label = labels[i]
        
        if label_mask is not None:
            mask = label_mask[i]
            length = mask.sum()
            pred = pred[:length]
            label = label[:length]
        
        # 过滤 ignore_index
        valid_mask = label != ignore_index
        if valid_mask.sum() == 0:
            continue
        
        pred_valid = pred[valid_mask]
        label_valid = label[valid_mask]
        
        # 计算 PER
        fper = compute_frame_per(pred, label, ignore_index)
        sper = compute_sequence_per(pred_valid, label_valid)
        sper_s = compute_smoothed_per(pred, label, ignore_index)
        
        fper_list.append(fper)
        sper_list.append(sper)
        sper_s_list.append(sper_s)
    
    return {
        "fper": np.mean(fper_list) if fper_list else 0.0,
        "sper": np.mean(sper_list) if sper_list else 0.0,
        "sper_s": np.mean(sper_s_list) if sper_s_list else 0.0,
    }
