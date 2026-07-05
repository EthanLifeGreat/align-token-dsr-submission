"""
Wav2Phoneme Factory Module

提供统一的接口供 src/train.py 调用
"""
from pathlib import Path
from typing import Tuple, List, TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments

from . import config as config_module
from .dataset import Wav2PhonemeDataset
from .collator import get_collator as get_collator_fn
from .model import Wav2PhonemeModel
from .model_metrics import METRIC_NAMES
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
METRIC_NAMES: List[str] = METRIC_NAMES


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

    train_dataset = Wav2PhonemeDataset(
        data_dir=data_dir,
        dataset=train_config.dataset,
        phone_dict_path=phone_dict_path,
        split="train",
    )

    eval_dataset = Wav2PhonemeDataset(
        data_dir=data_dir,
        dataset=train_config.dataset,
        phone_dict_path=phone_dict_path,
        split="dev",
    )

    # 限制 eval 数据集大小（解决 OOM 问题）
    eval_subset_size = getattr(train_config, 'eval_subset_size', None)
    if eval_subset_size is not None and len(eval_dataset) > eval_subset_size:
        import torch
        generator = torch.Generator().manual_seed(train_config.seed)
        indices = torch.randperm(len(eval_dataset), generator=generator)[:eval_subset_size].tolist()
        eval_dataset = torch.utils.data.Subset(eval_dataset, indices)

    return train_dataset, eval_dataset


# ============================================
# Collator
# ============================================

def get_collator(model_config, train_config):
    """
    创建数据 collator

    Args:
        model_config: 模型配置
        train_config: 训练配置

    Returns:
        Collator 实例
    """
    mode = getattr(model_config, 'mode', 'forced_alignment')
    kwargs = {}
    if mode == "ctc":
        kwargs["sil_id"] = int(getattr(model_config, "sil_id", 0))
        kwargs["remove_sil"] = bool(getattr(model_config, "ctc_remove_sil", True))
    elif mode in {"forced_alignment", "interpolate"}:
        kwargs["sil_id"] = int(getattr(model_config, "sil_id", 0))
    return get_collator_fn(mode, **kwargs)


# ============================================
# 模型
# ============================================

def get_model(
    model_config: "ModelConfig",
    train_config: "TrainConfig",
    train_dataset: Dataset,
) -> Wav2PhonemeModel:
    """
    创建模型

    Args:
        model_config: 模型配置
        train_config: 训练配置
        train_dataset: 训练数据集

    Returns:
        Wav2PhonemeModel 实例
    """
    return Wav2PhonemeModel(model_config)


# ============================================
# Trainer
# ============================================

def get_trainer(
    model: Wav2PhonemeModel,
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
        model_config: 模型配置
        train_config: 训练配置
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
    )
