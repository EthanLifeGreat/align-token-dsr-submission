"""
Wav2Phoneme 模块

将音频转换为帧级别的音素序列，基于 wav2vec 2.0 预训练模型。
"""

from src.wav2phoneme.config import ModelConfig
from src.train_utils.config import TrainConfig
from src.wav2phoneme.model import Wav2PhonemeModel
from src.wav2phoneme.dataset import Wav2PhonemeDataset, create_dataloader
from src.wav2phoneme.collator import ForcedAlignmentCollator, CTCCollator, InterpolateCollator, get_collator
from src.wav2phoneme.loss import forced_alignment_loss, ctc_loss, interpolate_loss

try:
    from src.wav2phoneme.metric import compute_frame_per, compute_sequence_per, compute_smoothed_per
except ImportError:  # pragma: no cover - optional metric dependency
    compute_frame_per = None
    compute_sequence_per = None
    compute_smoothed_per = None

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "Wav2PhonemeModel",
    "Wav2PhonemeDataset",
    "create_dataloader",
    "ForcedAlignmentCollator",
    "CTCCollator",
    "InterpolateCollator",
    "get_collator",
    "forced_alignment_loss",
    "ctc_loss",
    "interpolate_loss",
    "compute_frame_per",
    "compute_sequence_per",
    "compute_smoothed_per",
]
