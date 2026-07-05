"""
工具函数
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def load_phone_dict(path: str) -> Dict[str, int]:
    """
    加载音素字典

    Args:
        path: phone_dict.json 路径

    Returns:
        phoneme_name -> id 的映射字典
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        # phone_dict.json format: {"0": "sil", "1": "$0", ...}
        # Need to reverse to get {phoneme_name: id}
        id_to_phone = json.load(f)
    return {v: int(k) for k, v in id_to_phone.items()}


def load_id_to_phone(path: str) -> Dict[int, str]:
    """
    加载 id -> phoneme 的反向映射

    Args:
        path: phone_dict.json 路径

    Returns:
        id -> phoneme_name 的映射字典
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        # phone_dict.json format: {"0": "sil", "1": "$0", ...}
        id_to_phone = json.load(f)
    return {int(k): v for k, v in id_to_phone.items()}


def phoneme_to_id(phoneme_name: str, phone_dict: Dict[str, int]) -> int:
    """
    音素名转 ID
    
    Args:
        phoneme_name: 音素名称
        phone_dict: 音素字典
    
    Returns:
        音素 ID
    """
    return phone_dict.get(phoneme_name, phone_dict.get("sil", 0))


def id_to_phoneme(phoneme_id: int, id_to_phone_dict: Dict[int, str]) -> str:
    """
    ID 转音素名
    
    Args:
        phoneme_id: 音素 ID
        id_to_phone_dict: id -> phoneme 映射
    
    Returns:
        音素名称
    """
    return id_to_phone_dict.get(phoneme_id, "unk")


def downsample_phoneme(phoneme_ids: np.ndarray) -> np.ndarray:
    """
    下采样音素序列 (100fps -> 50fps)
    取偶数位置 [0, 2, 4, 6, ...]
    
    Args:
        phoneme_ids: [frames], 100fps 的音素 ID 序列
    
    Returns:
        [frames//2], 50fps 的音素 ID 序列
    """
    return phoneme_ids[::2].copy()


def deduplicate_phoneme(phoneme_ids: np.ndarray, remove_sil: bool = True, sil_id: int = 0) -> np.ndarray:
    """
    去重音素序列（CTC decode 风格）
    连续相同的音素只保留一个
    
    Args:
        phoneme_ids: 音素 ID 序列
        remove_sil: 是否移除 sil (ID=0)
        sil_id: sil 的 ID
    
    Returns:
        去重后的音素 ID 序列
    """
    if len(phoneme_ids) == 0:
        return phoneme_ids.copy()
    
    # 去重
    result = [phoneme_ids[0]]
    for i in range(1, len(phoneme_ids)):
        if phoneme_ids[i] != phoneme_ids[i-1]:
            result.append(phoneme_ids[i])
    
    result = np.array(result)
    
    # 移除 sil
    if remove_sil:
        result = result[result != sil_id]
    
    return result


def interpolate_phoneme(
    phoneme_ids: np.ndarray, 
    target_length: int,
    remove_sil: bool = True,
    sil_id: int = 0
) -> np.ndarray:
    """
    插值音素序列
    去重后插值到目标长度
    
    Args:
        phoneme_ids: 原始音素 ID 序列
        target_length: 目标长度
        remove_sil: 是否移除 sil
        sil_id: sil 的 ID
    
    Returns:
        插值后的音素 ID 序列
    """
    # 先去重
    dedup = deduplicate_phoneme(phoneme_ids, remove_sil=remove_sil, sil_id=sil_id)
    
    if len(dedup) == 0:
        return np.zeros(target_length, dtype=phoneme_ids.dtype)
    
    # 插值到目标长度
    indices = np.linspace(0, len(dedup) - 1, target_length)
    result = dedup[np.round(indices).astype(int)]
    
    return result


def load_phoneme_file(path: str) -> Tuple[List[str], np.ndarray]:
    """
    加载 .phone 文件
    
    文件格式：每行一个音素，格式为 "phoneme_name" 或 "phoneme_name start end"
    
    Args:
        path: .phone 文件路径
    
    Returns:
        phoneme_names: 音素名称列表
        phoneme_ids: 音素 ID 数组（需要外部传入 phone_dict 转换）
    """
    path = Path(path)
    phoneme_names = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            phoneme_names.append(parts[0])
    
    return phoneme_names


def parse_phoneme_file(path: str, phone_dict: Dict[str, int]) -> Tuple[List[str], np.ndarray]:
    """
    解析 .phone 文件并转换为 ID
    
    Args:
        path: .phone 文件路径
        phone_dict: 音素字典
    
    Returns:
        phoneme_names: 音素名称列表
        phoneme_ids: 音素 ID 数组
    """
    phoneme_names = load_phoneme_file(path)
    phoneme_ids = np.array([phoneme_to_id(name, phone_dict) for name in phoneme_names], dtype=np.int64)
    return phoneme_names, phoneme_ids
