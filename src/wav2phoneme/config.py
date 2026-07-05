"""
配置类定义
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml
from pathlib import Path
from transformers import PretrainedConfig

# 从 train_utils 导入通用 TrainConfig
from src.train_utils.config import TrainConfig, save_yaml, load_yaml


class ModelConfig(PretrainedConfig):
    """模型配置 (config.yaml) - 继承自 PretrainedConfig 以支持 HuggingFace 生态"""
    
    model_type = "wav2phoneme"
    
    def __init__(
        self,
        wav2vec_model_path: str = "TencentGameMate/chinese-wav2vec2-large",
        num_phonemes: int = 200,
        mode: str = "forced_alignment",
        phone_dict_path: str = "data/AISHELL-2/phone_dict.json",
        sil_id: int = 0,
        ctc_blank_id: int = 0,
        ctc_remove_sil: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wav2vec_model_path = wav2vec_model_path
        self.num_phonemes = num_phonemes
        self.mode = mode
        self.phone_dict_path = phone_dict_path
        self.sil_id = int(sil_id)
        self.ctc_blank_id = int(ctc_blank_id)
        self.ctc_remove_sil = bool(ctc_remove_sil)


# TrainConfig 从 train_utils.config 导入，此处仅保留 ModelConfig 相关函数


def save_model_config(config: ModelConfig, output_dir: Path) -> None:
    """保存模型配置"""
    save_yaml(config, Path(output_dir) / "config.yaml")


def load_model_config(path: Path) -> ModelConfig:
    """加载模型配置"""
    return load_yaml(ModelConfig, path)
