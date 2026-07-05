"""
Phoneme2Mel Factory Module

提供统一的接口供 src/train.py 调用
"""
from pathlib import Path
from typing import Tuple, List, TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments

from . import config as config_module
from .dataset import Phoneme2MelDataset
from .collator import Phoneme2MelCollator
from .model import Phoneme2MelModel
from src.train_utils.trainer import BaseTrainer
from src.train_utils.config import (
    load_train_config as _load_train_config,
    save_train_config as _save_train_config,
)

if TYPE_CHECKING:
    from .config import ModelConfig
    from src.train_utils.config import TrainConfig


# ============================================
# 模型配置
# ============================================

# 需要记录的额外指标名称
METRIC_NAMES: List[str] = ["loss_after_postnet", "loss_before_postnet"]


def _sync_model_vocab(model_config: "ModelConfig", num_phonemes: int = None) -> None:
    """根据真实音素表同步 num_phonemes / pad_id / vocab_size。"""
    if num_phonemes is not None:
        model_config.num_phonemes = num_phonemes
    model_config.pad_id = model_config.num_phonemes
    model_config.vocab_size = model_config.num_phonemes + 1
    model_config._validate_special_tokens()


# ============================================
# 配置加载/保存
# ============================================

def load_model_config(path: Path):
    """加载模型配置"""
    return config_module.load_model_config(path)


def load_train_config(path: Path):
    """加载训练配置"""
    return _load_train_config(path)


def save_model_config(config, output_dir: Path) -> None:
    """保存模型配置"""
    config_module.save_model_config(config, output_dir)


def save_train_config(config, output_dir: Path) -> None:
    """保存训练配置"""
    _save_train_config(config, output_dir)


# ============================================
# 数据集
# ============================================

def get_datasets(model_config: "ModelConfig", train_config: "TrainConfig", project_root: Path) -> Tuple[Dataset, Dataset]:
    """
    创建训练和验证数据集

    Args:
        model_config: 模型配置
        train_config: 训练配置
        project_root: 项目根目录

    Returns:
        (train_dataset, eval_dataset)
    """
    data_dir = str(project_root / train_config.data_dir)
    phone_dict_path = str(project_root / model_config.phone_dict_path)

    train_dataset = Phoneme2MelDataset(
        data_dir=data_dir,
        dataset=train_config.dataset,
        split='train',
        valid_frames_csv=getattr(train_config, 'valid_frames_csv', None),
        phone_dict=phone_dict_path,
        xvector_backend=model_config.xvector_backend,
        expected_xvector_dim=model_config.xvector_dim,
        expected_mel_dim=model_config.mel_dim,
        max_samples=getattr(train_config, "train_max_samples", None),
    )

    eval_dataset = Phoneme2MelDataset(
        data_dir=data_dir,
        dataset=train_config.dataset,
        split='val',  # 会自动映射到 'dev'
        valid_frames_csv=getattr(train_config, 'valid_frames_csv', None),
        phone_dict=phone_dict_path,
        xvector_backend=model_config.xvector_backend,
        expected_xvector_dim=model_config.xvector_dim,
        expected_mel_dim=model_config.mel_dim,
        max_samples=getattr(train_config, "eval_max_samples", None),
    )

    _sync_model_vocab(model_config, train_dataset.num_phonemes)

    return train_dataset, eval_dataset


# ============================================
# Collator
# ============================================

def get_collator(model_config: "ModelConfig", train_config: "TrainConfig"):
    """
    创建数据 collator

    Args:
        model_config: 模型配置
        train_config: 训练配置

    Returns:
        Phoneme2MelCollator 实例
    """
    _sync_model_vocab(model_config)
    return Phoneme2MelCollator(pad_id=model_config.pad_id)


# ============================================
# 模型
# ============================================

def get_model(
    model_config: "ModelConfig",
    train_config: "TrainConfig",
    train_dataset: Dataset,
) -> Phoneme2MelModel:
    """
    创建模型

    Args:
        model_config: 模型配置
        train_config: 训练配置
        train_dataset: 训练数据集（用于获取 num_phonemes）

    Returns:
        Phoneme2MelModel 实例
    """
    _sync_model_vocab(model_config, train_dataset.num_phonemes)

    return Phoneme2MelModel(model_config)


# ============================================
# Trainer
# ============================================

def get_trainer(
    model: Phoneme2MelModel,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    data_collator,
    model_config: "ModelConfig" = None,
    train_config: "TrainConfig" = None,
    **kwargs,
) -> BaseTrainer:
    """
    创建 Trainer

    Args:
        model: 模型
        args: TrainingArguments
        train_dataset: 训练数据集
        eval_dataset: 验证数据集
        data_collator: 数据 collator
        model_config: 模型配置（可选）
        train_config: 训练配置（可选）
        **kwargs: 其他参数

    Returns:
        BaseTrainer 实例
    """
    return BaseTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        metric_names=METRIC_NAMES,
        train_lengths=getattr(train_dataset, "lengths", None),
        eval_lengths=getattr(eval_dataset, "lengths", None),
        length_bucket_multiplier=getattr(train_config, "length_bucket_multiplier", 50),
        batch_size_unit=getattr(train_config, "batch_size_unit", "sample"),
        eval_batch_size_unit=getattr(args, "eval_batch_size_unit", None),
    )
