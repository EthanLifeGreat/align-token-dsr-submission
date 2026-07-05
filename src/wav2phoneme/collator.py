"""
Collator 定义 - 三种训练模式的 collate_fn
"""
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from src.wav2phoneme.utils import downsample_phoneme, deduplicate_phoneme, interpolate_phoneme


# Wav2Vec2 feature encoder config (same for TencentGameMate/chinese-wav2vec2-large)
WAV2VEC_CONV_KERNEL = [10, 3, 3, 3, 3, 2, 2]
WAV2VEC_CONV_STRIDE = [5, 2, 2, 2, 2, 2, 2]


def get_encoder_output_length(audio_length: Union[int, torch.LongTensor]) -> Union[int, torch.LongTensor]:
    """
    计算 Wav2Vec2 encoder 输出的帧数

    Args:
        audio_length: 音频样本数 (int or LongTensor)

    Returns:
        encoder 输出的帧数
    """
    def _conv_out_length(input_length, kernel_size, stride):
        return (input_length - kernel_size) // stride + 1

    for kernel_size, stride in zip(WAV2VEC_CONV_KERNEL, WAV2VEC_CONV_STRIDE):
        audio_length = _conv_out_length(audio_length, kernel_size, stride)
    return audio_length


def interpolate_to_length(phoneme_ids: np.ndarray, target_len: int) -> np.ndarray:
    """
    将 phoneme 序列插值到目标长度 (逐样本处理)

    Args:
        phoneme_ids: [T_phoneme], 原始音素 ID 序列
        target_len: 目标长度

    Returns:
        [target_len], 插值后的音素序列
    """
    if len(phoneme_ids) == target_len:
        return phoneme_ids

    # unsqueeze twice to get [1, 1, T] for interpolate
    phoneme_tensor = torch.from_numpy(phoneme_ids).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    interp = torch.nn.functional.interpolate(
        phoneme_tensor, size=target_len, mode='nearest'
    )
    return interp.squeeze(0).squeeze(0).long().numpy()


class ForcedAlignmentCollator:
    """
    Forced Alignment 模式的 Collator

    - 音频 padding
    - 音素插值到 Wav2Vec2 encoder 输出长度 (逐样本处理)
    - 返回：input_values, attention_mask, labels, label_mask
    """

    def __init__(self, sil_id: int = 0, ignore_index: int = -1):
        """
        Args:
            sil_id: sil 的音素 ID
            ignore_index: 用于 loss 计算的 ignore_index
        """
        self.sil_id = sil_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: List of dataset items

        Returns:
            {
                "input_values": Tensor,      # [B, T_audio]
                "attention_mask": Tensor,    # [B, T_audio]
                "labels": Tensor,            # [B, T_encoder]
                "label_mask": Tensor,        # [B, T_encoder]
            }
        """
        # 提取音频和音素
        audios = [item["audio"] for item in batch]
        phoneme_ids_list = [item["phoneme_ids"] for item in batch]

        # 音频 padding
        audio_tensors = [torch.from_numpy(audio) for audio in audios]
        input_values = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)

        # attention_mask
        attention_mask = torch.zeros_like(input_values, dtype=torch.long)
        audio_lengths = []
        for i, audio in enumerate(audio_tensors):
            attention_mask[i, :len(audio)] = 1
            audio_lengths.append(len(audio))

        # 逐样本计算 encoder 输出长度，并插值 phoneme 到该长度
        phoneme_interpolated = []
        for audio_len, phoneme_ids in zip(audio_lengths, phoneme_ids_list):
            target_len = get_encoder_output_length(audio_len)
            phoneme_interp = interpolate_to_length(phoneme_ids, target_len)
            phoneme_interpolated.append(phoneme_interp)

        # 音素 padding
        phoneme_tensors = [torch.from_numpy(p) for p in phoneme_interpolated]
        labels = pad_sequence(phoneme_tensors, batch_first=True, padding_value=self.ignore_index)

        # label_mask
        label_mask = torch.zeros_like(labels, dtype=torch.long)
        for i, p in enumerate(phoneme_tensors):
            label_mask[i, :len(p)] = 1

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_mask": label_mask,
        }


class CTCCollator:
    """
    CTC 模式的 Collator
    
    - 音频 padding
    - 音素去重
    - 可选移除 sil；blank 由模型配置单独定义，不再复用 sil
    - 返回：input_values, attention_mask, labels, label_lengths
    """
    
    def __init__(self, sil_id: int = 0, remove_sil: bool = True):
        """
        Args:
            sil_id: sil 的音素 ID
            remove_sil: 是否在 CTC target 中移除 sil
        """
        self.sil_id = sil_id
        self.remove_sil = bool(remove_sil)
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: List of dataset items
        
        Returns:
            {
                "input_values": Tensor,
                "attention_mask": Tensor,
                "labels": Tensor,            # [B, T_phoneme_unique]
                "label_lengths": Tensor,     # [B]
            }
        """
        # 提取音频和音素
        audios = [item["audio"] for item in batch]
        phoneme_ids_list = [item["phoneme_ids"] for item in batch]
        
        # 音频 padding
        audio_tensors = [torch.from_numpy(audio) for audio in audios]
        input_values = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
        
        # attention_mask
        attention_mask = torch.zeros_like(input_values, dtype=torch.long)
        for i, audio in enumerate(audio_tensors):
            attention_mask[i, :len(audio)] = 1
        
        # 音素去重；A1 需要保留显式 blank，并可选择保留真实 sil
        phoneme_dedup = [deduplicate_phoneme(p, remove_sil=self.remove_sil, sil_id=self.sil_id)
                        for p in phoneme_ids_list]
        
        # 音素 padding (用 0 作为 padding，CTC 会自动处理)
        phoneme_tensors = [torch.from_numpy(p) for p in phoneme_dedup]
        labels = pad_sequence(phoneme_tensors, batch_first=True, padding_value=0)
        
        # label_lengths
        label_lengths = torch.tensor([len(p) for p in phoneme_dedup], dtype=torch.long)
        
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_lengths": label_lengths,
        }


class InterpolateCollator:
    """
    Interpolate 模式的 Collator
    
    - 音频 padding
    - 音素去重、移除 sil、插值到原长度
    - 返回同 ForcedAlignmentCollator
    """
    
    def __init__(self, sil_id: int = 0, ignore_index: int = -1):
        """
        Args:
            sil_id: sil 的音素 ID
            ignore_index: 用于 loss 计算的 ignore_index
        """
        self.sil_id = sil_id
        self.ignore_index = ignore_index
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: List of dataset items
        
        Returns:
            {
                "input_values": Tensor,
                "attention_mask": Tensor,
                "labels": Tensor,
                "label_mask": Tensor,
            }
        """
        # 提取音频和音素
        audios = [item["audio"] for item in batch]
        phoneme_ids_list = [item["phoneme_ids"] for item in batch]
        
        # 音频 padding
        audio_tensors = [torch.from_numpy(audio) for audio in audios]
        input_values = pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
        
        # attention_mask
        attention_mask = torch.zeros_like(input_values, dtype=torch.long)
        for i, audio in enumerate(audio_tensors):
            attention_mask[i, :len(audio)] = 1
        
        # 音素下采样后的长度
        target_lengths = [len(p) // 2 for p in phoneme_ids_list]
        max_target_len = max(target_lengths) if target_lengths else 0
        
        # 音素去重、移除 sil、插值到下采样后的长度
        phoneme_interpolated = [
            interpolate_phoneme(p, target_len, remove_sil=True, sil_id=self.sil_id)
            for p, target_len in zip(phoneme_ids_list, target_lengths)
        ]
        
        # 音素 padding
        phoneme_tensors = [torch.from_numpy(p) for p in phoneme_interpolated]
        labels = pad_sequence(phoneme_tensors, batch_first=True, padding_value=self.ignore_index)
        
        # label_mask
        label_mask = torch.zeros_like(labels, dtype=torch.long)
        for i, p in enumerate(phoneme_tensors):
            label_mask[i, :len(p)] = 1
        
        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": labels,
            "label_mask": label_mask,
        }


def get_collator(mode: str, **kwargs):
    """
    根据模式获取 Collator
    
    Args:
        mode: "forced_alignment", "ctc", "interpolate"
        **kwargs: Collator 参数
    
    Returns:
        Collator 实例
    """
    mode = mode.lower()
    
    if mode == "forced_alignment":
        return ForcedAlignmentCollator(**kwargs)
    elif mode == "ctc":
        return CTCCollator(**kwargs)
    elif mode == "interpolate":
        return InterpolateCollator(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'forced_alignment', 'ctc', or 'interpolate'")
