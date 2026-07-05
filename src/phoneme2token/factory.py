"""Factory hooks for the unified training script."""
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

from torch.utils.data import Dataset
from transformers import TrainingArguments

from src.train_utils.config import (
    load_train_config as _load_train_config,
    save_train_config as _save_train_config,
)
from src.train_utils.trainer import BaseTrainer

from . import config as config_module
from .collator import Phoneme2TokenCollator
from .dataset import Phoneme2TokenDataset
from .model import Phoneme2TokenModel
from .model_metrics import METRIC_NAMES

if TYPE_CHECKING:
    from src.train_utils.config import TrainConfig
    from .config import ModelConfig


def _sync_model_vocab(model_config: "ModelConfig", num_phonemes: int | None = None) -> None:
    if num_phonemes is not None:
        model_config.num_phonemes = int(num_phonemes)
    model_config.phoneme_pad_id = model_config.num_phonemes
    model_config.phoneme_vocab_size = model_config.num_phonemes + 1
    model_config.token_pad_id = model_config.speech_vocab_size
    model_config.bos_id = model_config.speech_vocab_size + 1
    model_config.token_input_vocab_size = model_config.bos_id + 1
    model_config.token_output_vocab_size = model_config.speech_vocab_size + 1
    model_config._validate()


def load_model_config(path: Path):
    return config_module.load_model_config(path)


def save_model_config(config, output_dir: Path) -> None:
    config_module.save_model_config(config, output_dir)


def load_train_config(path: Path):
    return _load_train_config(path)


def save_train_config(config, output_dir: Path) -> None:
    _save_train_config(config, output_dir)


def get_datasets(
    model_config: "ModelConfig",
    train_config: "TrainConfig",
    project_root: Path,
) -> Tuple[Dataset, Dataset]:
    data_dir = str(project_root / train_config.data_dir)
    phone_dict_path = str(project_root / model_config.phone_dict_path)
    valid_frames_csv = getattr(train_config, "valid_frames_csv", None)

    train_dataset = Phoneme2TokenDataset(
        data_dir=data_dir,
        dataset=train_config.dataset,
        split="train",
        phone_dict=phone_dict_path,
        valid_frames_csv=valid_frames_csv,
        token_dir_name=model_config.token_dir_name,
        xvector_backend=model_config.xvector_backend,
        speech_vocab_size=model_config.speech_vocab_size,
        token_pad_id=model_config.token_pad_id,
        bos_id=model_config.bos_id,
        length_pad_margin=model_config.length_pad_margin,
        frontend_mode=model_config.frontend_mode,
        max_samples=getattr(train_config, "train_max_samples", None),
        wav2phoneme_ckpt=model_config.wav2phoneme_ckpt,
    )
    eval_dataset = Phoneme2TokenDataset(
        data_dir=data_dir,
        dataset=train_config.dataset,
        split="val",
        phone_dict=phone_dict_path,
        valid_frames_csv=valid_frames_csv,
        token_dir_name=model_config.token_dir_name,
        xvector_backend=model_config.xvector_backend,
        speech_vocab_size=model_config.speech_vocab_size,
        token_pad_id=model_config.token_pad_id,
        bos_id=model_config.bos_id,
        length_pad_margin=model_config.length_pad_margin,
        frontend_mode=model_config.frontend_mode,
        max_samples=getattr(train_config, "eval_max_samples", None),
        wav2phoneme_ckpt=model_config.wav2phoneme_ckpt,
    )
    _sync_model_vocab(model_config, train_dataset.num_phonemes)
    return train_dataset, eval_dataset


def get_collator(model_config: "ModelConfig", train_config: "TrainConfig"):
    _sync_model_vocab(model_config)
    return Phoneme2TokenCollator(
        phoneme_pad_id=model_config.phoneme_pad_id,
        token_pad_id=model_config.token_pad_id,
        ignore_index=-100,
    )


def get_model(
    model_config: "ModelConfig",
    train_config: "TrainConfig",
    train_dataset: Dataset,
) -> Phoneme2TokenModel:
    _sync_model_vocab(model_config, train_dataset.num_phonemes)
    return Phoneme2TokenModel(model_config)


def get_trainer(
    model: Phoneme2TokenModel,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    data_collator,
    model_config: "ModelConfig" = None,
    train_config: "TrainConfig" = None,
    **kwargs,
) -> BaseTrainer:
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
        batch_size_unit="sample",
        eval_batch_size_unit="sample",
    )
