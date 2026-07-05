"""Configuration for phoneme2token."""
from pathlib import Path
from typing import Optional

from transformers import PretrainedConfig

from src.train_utils.config import load_yaml, save_yaml


class ModelConfig(PretrainedConfig):
    """Model architecture config for phoneme2token."""

    model_type = "phoneme2token"

    def __init__(
        self,
        xvector_dim: int = 192,
        num_phonemes: int = 200,
        phoneme_pad_id: Optional[int] = None,
        phoneme_vocab_size: Optional[int] = None,
        speech_vocab_size: int = 6561,
        token_pad_id: Optional[int] = None,
        bos_id: Optional[int] = None,
        tie_word_embeddings: bool = True,
        d_model: int = 384,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_position_embeddings: int = 4096,
        phone_dict_path: str = "data/AISHELL-2-95_5/phone_dict.json",
        token_dir_name: str = "s3tokenizer_v2_25hz",
        xvector_backend: str = "eresnet2v2",
        length_pad_margin: int = 2,
        frontend_mode: str = "precomputed",
        wav2phoneme_ckpt: Optional[str] = None,
        frontend_cache_dir: Optional[str] = None,
        frontend_repeat_factor: int = 2,
        frontend_blank_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.xvector_dim = int(xvector_dim)
        self.num_phonemes = int(num_phonemes)
        self.modeling_mode = "decoder_only"
        self.phoneme_pad_id = self.num_phonemes if phoneme_pad_id is None else int(phoneme_pad_id)
        self.phoneme_vocab_size = (
            self.num_phonemes + 1
            if phoneme_vocab_size is None
            else int(phoneme_vocab_size)
        )
        self.speech_vocab_size = int(speech_vocab_size)
        self.token_pad_id = self.speech_vocab_size if token_pad_id is None else int(token_pad_id)
        self.bos_id = self.speech_vocab_size + 1 if bos_id is None else int(bos_id)
        self.token_output_vocab_size = self.speech_vocab_size + 1
        self.token_input_vocab_size = self.bos_id + 1
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.max_position_embeddings = int(max_position_embeddings)
        self.phone_dict_path = str(phone_dict_path)
        self.token_dir_name = str(token_dir_name)
        self.xvector_backend = str(xvector_backend)
        self.length_pad_margin = int(length_pad_margin)
        self.frontend_mode = str(frontend_mode)
        self.wav2phoneme_ckpt = None if wav2phoneme_ckpt is None else str(wav2phoneme_ckpt)
        self.frontend_cache_dir = (
            None if frontend_cache_dir is None else str(frontend_cache_dir)
        )
        self.frontend_repeat_factor = int(frontend_repeat_factor)
        self.frontend_blank_id = (
            None if frontend_blank_id is None else int(frontend_blank_id)
        )
        self._validate()

    def _validate(self) -> None:
        if self.num_phonemes <= 0:
            raise ValueError("num_phonemes must be positive")
        if self.phoneme_pad_id < self.num_phonemes:
            raise ValueError("phoneme_pad_id must not overlap real phoneme ids")
        if self.phoneme_vocab_size <= self.phoneme_pad_id:
            raise ValueError("phoneme_vocab_size must include phoneme_pad_id")
        if self.speech_vocab_size <= 0:
            raise ValueError("speech_vocab_size must be positive")
        if self.token_pad_id != self.speech_vocab_size:
            raise ValueError("token_pad_id must equal speech_vocab_size")
        if self.bos_id != self.speech_vocab_size + 1:
            raise ValueError("bos_id must equal speech_vocab_size + 1")
        if self.modeling_mode != "decoder_only":
            raise ValueError("phoneme2token submission config only supports decoder_only")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.length_pad_margin < 0:
            raise ValueError("length_pad_margin must be non-negative")
        if self.frontend_mode not in {"precomputed", "wav2phoneme_ctc_frameargmax"}:
            raise ValueError(
                "frontend_mode must be 'precomputed' or 'wav2phoneme_ctc_frameargmax'"
            )
        if self.frontend_mode == "wav2phoneme_ctc_frameargmax" and not self.wav2phoneme_ckpt:
            raise ValueError("wav2phoneme_ckpt is required for frameargmax frontend mode")
        if self.frontend_repeat_factor <= 0:
            raise ValueError("frontend_repeat_factor must be positive")


def save_model_config(config: ModelConfig, output_dir: Path) -> None:
    save_yaml(config, Path(output_dir) / "config.yaml")


def load_model_config(path: Path) -> ModelConfig:
    return load_yaml(ModelConfig, path)
