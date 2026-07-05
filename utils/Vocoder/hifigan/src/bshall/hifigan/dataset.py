from pathlib import Path
import math
import random
import numpy as np
import pandas
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchaudio

from .mel_spec import LogMelSpectrogram


class MelDataset(Dataset):
    def __init__(
        self,
        id2path_df: pandas.DataFrame,
        wav_data_dir: str,
        segment_length: int,
        sample_rate: int,
        hop_length: int,
        train: bool = True,
        mel_data_dir: str = None,
        input_mel_source: str = "wav",
        disable_augment: bool = False,
    ):
        self.id2path_df = id2path_df
        self.wav_data_dir = wav_data_dir
        self.mel_data_dir = mel_data_dir

        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.train = train
        self.input_mel_source = input_mel_source
        self.disable_augment = disable_augment

        self.logmel = LogMelSpectrogram()

    @staticmethod
    def _resolve_dir(row: pandas.Series, column: str, fallback_dir: str = None):
        if fallback_dir is not None:
            return Path(fallback_dir)

        if column not in row:
            return None

        value = row[column]
        if pandas.isna(value):
            return None

        value = str(value)
        if not value:
            return None

        return Path(value)

    @staticmethod
    def _resolve_data_path(
        row: pandas.Series,
        base_dir: Path,
        path_column: str,
        speaker_column: str,
        item_id_column: str,
        suffix: str,
    ):
        if path_column in row and not pandas.isna(row[path_column]):
            item_path = Path(str(row[path_column]))
            if item_path.suffix != suffix:
                item_path = item_path.with_suffix(suffix)
            if item_path.is_absolute() or base_dir is None:
                return item_path
            return base_dir / item_path

        if speaker_column not in row or item_id_column not in row:
            raise KeyError(
                f"Expected either '{path_column}' or both '{speaker_column}' and '{item_id_column}' columns"
            )

        if base_dir is None:
            raise ValueError(
                f"Missing base directory for {path_column}; provide dataset dir or '{path_column}' column"
            )

        speaker = str(row[speaker_column])
        item_id = str(row[item_id_column])
        return base_dir / speaker / f"{item_id}{suffix}"

    def __len__(self):
        return len(self.id2path_df)

    def __getitem__(self, index):
        row = self.id2path_df.iloc[index]
        wav_data_dir = self._resolve_dir(row, "wav_data_dir", self.wav_data_dir)
        wav_path = self._resolve_data_path(
            row=row,
            base_dir=wav_data_dir,
            path_column="wav_path",
            speaker_column="spk_id",
            item_id_column="wav_id",
            suffix=".wav",
        )

        info = torchaudio.info(str(wav_path))
        if info.sample_rate != self.sample_rate:
            raise ValueError(
                f"Sample rate {info.sample_rate} doesn't match target of {self.sample_rate}"
            )

        use_precomputed_mel = self.input_mel_source == "precomputed"
        if use_precomputed_mel:
            mel_data_dir = self._resolve_dir(row, "mel_data_dir", self.mel_data_dir)
            if mel_data_dir is None:
                raise ValueError(
                    "mel_data_dir is required when input_mel_source='precomputed'"
                )

            mel_path = self._resolve_data_path(
                row=row,
                base_dir=mel_data_dir,
                path_column="mel_path",
                speaker_column="spk_id",
                item_id_column="wav_id",
                suffix=".npy",
            )
            src_logmel = torch.from_numpy(np.load(mel_path))
            src_logmel = src_logmel.unsqueeze(0).transpose(1, 2)

            mel_frames_per_segment = math.ceil(self.segment_length / self.hop_length)
            mel_diff = src_logmel.size(-1) - mel_frames_per_segment if self.train else 0
            mel_offset = random.randint(0, max(mel_diff, 0))

            frame_offset = self.hop_length * mel_offset
        else:
            frame_diff = info.num_frames - self.segment_length
            frame_offset = random.randint(0, max(frame_diff, 0))

        wav, _ = torchaudio.load(
            filepath=str(wav_path),
            frame_offset=frame_offset if self.train else 0,
            num_frames=self.segment_length if self.train else -1,
        )

        if wav.size(-1) < self.segment_length:
            wav = F.pad(wav, (0, self.segment_length - wav.size(-1)))

        if not self.disable_augment and self.train:
            gain = random.random() * (0.99 - 0.4) + 0.4
            flip = -1 if random.random() > 0.5 else 1
            wav = flip * gain * wav / max(wav.abs().max(), 1e-5)

        tgt_logmel = self.logmel(wav.unsqueeze(0)).squeeze(0)

        if use_precomputed_mel:
            if self.train:
                src_logmel = src_logmel[
                    :, :, mel_offset : mel_offset + mel_frames_per_segment
                ]

            if src_logmel.size(-1) < mel_frames_per_segment:
                src_logmel = F.pad(
                    src_logmel,
                    (0, mel_frames_per_segment - src_logmel.size(-1)),
                    "constant",
                    src_logmel.min(),
                )
        else:
            src_logmel = tgt_logmel.clone()

        return wav, src_logmel, tgt_logmel
