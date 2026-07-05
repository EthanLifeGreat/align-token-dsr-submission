"""
配置类定义
"""
from typing import Optional
from pathlib import Path
from transformers import PretrainedConfig

from src.train_utils.config import save_yaml, load_yaml


class ModelConfig(PretrainedConfig):
    """模型配置 (config.yaml) - 继承自 PretrainedConfig 以支持 HuggingFace 生态"""

    model_type = "phoneme2mel"

    def __init__(
        self,
        xvector_dim: int = 192,
        num_phonemes: int = 200,
        vocab_size: Optional[int] = None,
        pad_id: Optional[int] = None,
        d_model: int = 384,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_position_embeddings: int = 4096,
        backbone_type: str = "rope",
        phoneme_embed_dim: Optional[int] = None,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        mel_dim: int = 128,
        postnet_type: str = "tacotron",
        postnet_n_layers: int = 5,
        postnet_kernel_size: int = 5,
        postnet_dim: int = 512,
        postnet_dropout: Optional[float] = None,
        phone_dict_path: str = "data/AISHELL-2-95_5/phone_dict.json",
        xvector_backend: str = "eresnet2v2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.xvector_dim = int(xvector_dim)
        self.num_phonemes = int(num_phonemes)
        self.pad_id = self.num_phonemes if pad_id is None else int(pad_id)
        self.vocab_size = (self.num_phonemes + 1) if vocab_size is None else int(vocab_size)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.max_position_embeddings = int(max_position_embeddings)
        self.backbone_type = str(backbone_type)
        self.phoneme_embed_dim = (
            self.d_model if phoneme_embed_dim is None else int(phoneme_embed_dim)
        )
        self.activation = str(activation)
        self.layer_norm_eps = float(layer_norm_eps)
        self.mel_dim = int(mel_dim)
        self.postnet_type = str(postnet_type)
        self.postnet_n_layers = int(postnet_n_layers)
        self.postnet_kernel_size = int(postnet_kernel_size)
        self.postnet_dim = int(postnet_dim)
        self.postnet_dropout = (
            self.dropout if postnet_dropout is None else float(postnet_dropout)
        )
        self.phone_dict_path = str(phone_dict_path)
        self.xvector_backend = str(xvector_backend)

        self._validate_special_tokens()
        self._validate_architecture()

    def _validate_special_tokens(self) -> None:
        """确保 padding token 与真实音素 ID 不冲突。"""
        if self.num_phonemes <= 0:
            raise ValueError(f"num_phonemes must be positive, got {self.num_phonemes}")

        if self.vocab_size <= self.num_phonemes:
            raise ValueError(
                "vocab_size must be larger than num_phonemes to reserve padding space, "
                f"got vocab_size={self.vocab_size}, num_phonemes={self.num_phonemes}"
            )

        if not (0 <= self.pad_id < self.vocab_size):
            raise ValueError(
                f"pad_id must be within [0, vocab_size), got pad_id={self.pad_id}, "
                f"vocab_size={self.vocab_size}"
            )

        if self.pad_id < self.num_phonemes:
            raise ValueError(
                f"pad_id overlaps with phoneme ids [0, {self.num_phonemes - 1}], "
                f"got pad_id={self.pad_id}"
            )

    def _validate_architecture(self) -> None:
        if self.xvector_dim != 192:
            raise ValueError(f"xvector_dim must be 192 for the new phoneme2mel contract, got {self.xvector_dim}")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.backbone_type not in {"rope", "legacy"}:
            raise ValueError(
                f"backbone_type must be 'rope' or 'legacy', got {self.backbone_type!r}"
            )
        if self.phoneme_embed_dim <= 0:
            raise ValueError("phoneme_embed_dim must be positive")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError(
                f"activation must be 'relu' or 'gelu', got {self.activation!r}"
            )
        if self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive")
        if self.backbone_type == "legacy" and self.nhead % 2 != 0:
            raise ValueError("legacy backbone requires an even nhead")
        if self.postnet_type not in {"tacotron", "legacy"}:
            raise ValueError(
                f"postnet_type must be 'tacotron' or 'legacy', got {self.postnet_type!r}"
            )
        if not (0.0 <= self.postnet_dropout < 1.0):
            raise ValueError(
                f"postnet_dropout must be within [0, 1), got {self.postnet_dropout}"
            )


# TrainConfig 从 train_utils.config 导入，此处仅保留 ModelConfig 相关函数


def save_model_config(config: ModelConfig, output_dir: Path) -> None:
    """保存模型配置"""
    save_yaml(config, Path(output_dir) / "config.yaml")


def load_model_config(path: Path) -> ModelConfig:
    """加载模型配置"""
    return load_yaml(ModelConfig, path)
