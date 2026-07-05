"""
数据集类定义
"""
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torchaudio

from src.wav2phoneme.utils import load_phone_dict, parse_phoneme_file


class Wav2PhonemeDataset(torch.utils.data.Dataset):
    """
    Wav2Phoneme 数据集
    
    数据目录结构：
        data_dir/
        └── {dataset}/
            ├── meta.csv
            ├── has_phoneme.csv
            ├── wav/{spk_id}/{wav_id}.wav
            └── phoneme/{spk_id}/{wav_id}.phone
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset: str,
        phone_dict_path: str,
        split: Optional[str] = None,
        sample_rate: int = 16000,
        use_valid_frames: bool = True,
    ):
        """
        Args:
            data_dir: 数据根目录 (e.g., "data/")
            dataset: 数据集名称 ("AISHELL-2", "CSMSC", "CDSD")
            phone_dict_path: phone_dict.json 路径
            split: 数据划分 ("train", "dev", "test", None 表示全部)
            sample_rate: 目标采样率
            use_valid_frames: 是否使用 valid_frames.csv 筛选（帧数差异 <= 2%）
        """
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.dataset_dir = self.data_dir / dataset
        self.split = split
        self.sample_rate = sample_rate
        self.use_valid_frames = use_valid_frames
        
        # 加载音素字典
        self.phone_dict = load_phone_dict(phone_dict_path)
        
        # 加载 meta.csv
        self.meta_path = self.dataset_dir / "meta.csv"
        self.has_phoneme_path = self.dataset_dir / "has_phoneme.csv"
        self.valid_frames_path = self.dataset_dir / "valid_frames.csv"
        
        # 加载筛选文件
        self.valid_samples = self._load_valid_samples()
        
        # 加载 meta.csv 并筛选
        self.meta_data = self._load_meta()
        
        print(f"[{dataset}] Loaded {len(self.meta_data)} samples" + 
              (f" (split={split})" if split else ""))
    
    def _load_valid_samples(self) -> Dict[str, bool]:
        """
        加载有效样本筛选文件
        
        优先使用 valid_frames.csv（帧数差异 <= 2%），
        如果不存在则使用 has_phoneme.csv
        """
        valid_samples = {}
        
        if self.use_valid_frames and self.valid_frames_path.exists():
            # 使用 valid_frames.csv
            with open(self.valid_frames_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    valid_samples[row["wav_id"]] = row["is_valid"] == "1"
            print(f"[{self.dataset}] Using valid_frames.csv for filtering")
        elif self.has_phoneme_path.exists():
            # 回退到 has_phoneme.csv
            with open(self.has_phoneme_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    valid_samples[row["wav_id"]] = row["has_phoneme"] == "1"
            print(f"[{self.dataset}] Using has_phoneme.csv for filtering (fallback)")
        
        return valid_samples
    
    def _load_meta(self) -> List[Dict]:
        """加载 meta.csv 并筛选"""
        meta_data = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                wav_id = row["wav_id"]
                
                # 筛选有效样本
                if self.valid_samples and not self.valid_samples.get(wav_id, False):
                    continue
                
                # 筛选 split
                if self.split is not None and row.get("split", "") != self.split:
                    continue
                
                meta_data.append({
                    "wav_id": wav_id,
                    "spk_id": row["spk_id"],
                    "text": row.get("text", ""),
                    "split": row.get("split", ""),
                })
        return meta_data
    
    def __len__(self) -> int:
        return len(self.meta_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                "wav_id": str,
                "spk_id": str,
                "audio": np.ndarray,  # [samples], 16kHz
                "phoneme_ids": np.ndarray,  # [frames]
                "phoneme_names": List[str],
            }
        """
        meta = self.meta_data[idx]
        wav_id = meta["wav_id"]
        spk_id = meta["spk_id"]
        
        # 加载音频
        audio = self._load_audio(wav_id, spk_id)
        
        # 加载音素
        phoneme_names, phoneme_ids = self._load_phoneme(wav_id, spk_id)
        
        return {
            "wav_id": wav_id,
            "spk_id": spk_id,
            "audio": audio,
            "phoneme_ids": phoneme_ids,
            "phoneme_names": phoneme_names,
        }
    
    def _load_audio(self, wav_id: str, spk_id: str) -> np.ndarray:
        """
        加载音频文件
        
        路径: {data_dir}/{dataset}/wav/{spk_id}/{wav_id}.wav
        """
        wav_path = self.dataset_dir / "wav" / spk_id / f"{wav_id}.wav"
        
        # 尝试不同扩展名
        if not wav_path.exists():
            wav_path = self.dataset_dir / "wav" / spk_id / f"{wav_id}.WAV"
        
        waveform, sr = torchaudio.load(str(wav_path))
        
        # 重采样
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # 返回 numpy array
        return waveform.squeeze(0).numpy()
    
    def _load_phoneme(self, wav_id: str, spk_id: str) -> Tuple[List[str], np.ndarray]:
        """
        加载音素文件
        
        路径: {data_dir}/{dataset}/phoneme/{spk_id}/{wav_id}.phone
        """
        phone_path = self.dataset_dir / "phoneme" / spk_id / f"{wav_id}.phone"
        
        if not phone_path.exists():
            # 返回空
            return [], np.array([], dtype=np.int64)
        
        phoneme_names, phoneme_ids = parse_phoneme_file(str(phone_path), self.phone_dict)
        return phoneme_names, phoneme_ids


def create_dataloader(
    dataset: Wav2PhonemeDataset,
    collate_fn: Callable,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    创建 DataLoader
    
    Args:
        dataset: 数据集
        collate_fn: collate 函数
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否 pin memory
    
    Returns:
        DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
