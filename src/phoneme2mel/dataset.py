"""
Dataset for the 192d-compatible phoneme2mel training pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def canonical_wav_id(wav_id: object) -> str:
    value = str(wav_id).strip()
    if value.isdigit():
        return str(int(value))
    return value


def format_wav_id_path(wav_id: object) -> str:
    value = str(wav_id).strip()
    if value.isdigit():
        return f"{int(value):06d}"
    return value


def load_phoneme_ids(phone_path: Path, phone_to_id: dict[str, int]) -> list[int]:
    with phone_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    phonemes = [line.strip() for line in lines[:-1]]
    sil_id = phone_to_id.get("sil", 0)
    return [phone_to_id.get(phoneme, sil_id) for phoneme in phonemes]


@dataclass(frozen=True)
class Phoneme2MelRecord:
    wav_id: str
    wav_id_path: str
    spk_id: str
    phone_path: Path
    mel_path: Path
    xvector_path: Path
    phone_frames: int


class Phoneme2MelDataset(Dataset):
    """
    Loads phoneme, mel, and xvector for 192d-compatible phoneme2mel training.
    """

    def __init__(
        self,
        data_dir: str,
        dataset: str = "CSMSC",
        split: str = "train",
        valid_frames_csv: Optional[str] = None,
        phone_dict: Optional[str] = None,
        xvector_backend: str = "eresnet2v2",
        expected_xvector_dim: int = 192,
        expected_mel_dim: int = 128,
        max_samples: Optional[int] = None,
        validation_samples: int = 8,
    ):
        self.data_dir = Path(data_dir) / dataset
        self.dataset = dataset
        self.split = "dev" if split == "val" else split
        self.xvector_backend = str(xvector_backend)
        self.expected_xvector_dim = int(expected_xvector_dim)
        self.expected_mel_dim = int(expected_mel_dim)
        self.max_samples = None if max_samples is None else int(max_samples)

        meta = pd.read_csv(self.data_dir / "meta.csv")
        if self.split in {"train", "dev", "test"}:
            meta = meta[meta["split"] == self.split]
        elif self.split != "all":
            raise ValueError(f"Unsupported split={split}")

        valid_keys = self._load_valid_keys(valid_frames_csv)
        if valid_keys is not None:
            meta = meta[
                meta.apply(
                    lambda row: (str(row["spk_id"]).strip(), canonical_wav_id(row["wav_id"])) in valid_keys,
                    axis=1,
                )
            ]

        if self.max_samples is not None:
            meta = meta.iloc[: self.max_samples]
        meta = meta.reset_index(drop=True)
        if meta.empty:
            raise ValueError(f"No samples found for {dataset}/{self.split}")

        self.phone_to_id = self._load_phone_dict(phone_dict)
        self.num_phonemes = max(self.phone_to_id.values()) + 1

        frame_info = self._load_frame_info()
        self.records = self._build_records(meta, frame_info)
        self.lengths = [record.phone_frames for record in self.records]

        self._validate_contract(num_samples=validation_samples)
        print(
            f"Loaded {len(self.records)} samples for {dataset}/{self.split}; "
            f"xvector_backend={self.xvector_backend}"
        )

    def _load_valid_keys(self, valid_frames_csv: Optional[str]) -> Optional[set[tuple[str, str]]]:
        if not valid_frames_csv:
            return None
        path = Path(valid_frames_csv)
        if not path.is_absolute():
            path = self.data_dir / path
        valid_df = pd.read_csv(path)
        valid_keys: set[tuple[str, str]] = set()
        for row in valid_df.itertuples(index=False):
            if hasattr(row, "is_valid") and str(getattr(row, "is_valid")).strip() in {"0", "false", "False"}:
                continue
            valid_keys.add((str(getattr(row, "spk_id")).strip(), canonical_wav_id(getattr(row, "wav_id"))))
        return valid_keys

    def _load_frame_info(self) -> dict[tuple[str, str], int]:
        candidate_csvs = [
            self.data_dir / "frame_diff_p2m.csv",
            self.data_dir / "frame_diff_p2t.csv",
        ]
        frame_csv = None
        frame_df = None
        for candidate in candidate_csvs:
            try:
                frame_df = pd.read_csv(candidate)
                frame_csv = candidate
                break
            except FileNotFoundError:
                continue
        if frame_df is None or frame_csv is None:
            raise FileNotFoundError(
                f"No frame diff csv found under {self.data_dir}; expected one of {candidate_csvs}"
            )
        if "phone_frames" not in frame_df.columns:
            raise ValueError(f"phone_frames column missing in {frame_csv}")

        frame_info: dict[tuple[str, str], int] = {}
        for row in frame_df.itertuples(index=False):
            if hasattr(row, "is_valid") and str(getattr(row, "is_valid")).strip() in {"0", "false", "False"}:
                continue
            spk_id = str(getattr(row, "spk_id")).strip()
            wav_id = canonical_wav_id(getattr(row, "wav_id"))
            frame_info[(spk_id, wav_id)] = int(getattr(row, "phone_frames"))
        return frame_info

    def _load_phone_dict(self, phone_dict: Optional[str]) -> dict[str, int]:
        if phone_dict is None:
            phone_dict_path = self.data_dir / "phone_dict.json"
            try:
                with phone_dict_path.open("r", encoding="utf-8") as f:
                    phone_id_to_name = json.load(f)
            except FileNotFoundError:
                phone_dict_path = Path("data/AISHELL-2-95_5/phone_dict.json")
                with phone_dict_path.open("r", encoding="utf-8") as f:
                    phone_id_to_name = json.load(f)
        else:
            phone_dict_path = Path(phone_dict)
            with phone_dict_path.open("r", encoding="utf-8") as f:
                phone_id_to_name = json.load(f)
        return {name: int(idx) for idx, name in phone_id_to_name.items()}

    def _build_records(
        self,
        meta: pd.DataFrame,
        frame_info: dict[tuple[str, str], int],
    ) -> list[Phoneme2MelRecord]:
        records: list[Phoneme2MelRecord] = []
        for row in meta.itertuples(index=False):
            wav_id = canonical_wav_id(row.wav_id)
            wav_id_path = format_wav_id_path(row.wav_id)
            spk_id = str(row.spk_id).strip()
            key = (spk_id, wav_id)
            if key not in frame_info:
                raise KeyError(f"Missing phone_frames for {self.dataset} sample {key}")
            records.append(
                Phoneme2MelRecord(
                    wav_id=wav_id,
                    wav_id_path=wav_id_path,
                    spk_id=spk_id,
                    phone_path=self.data_dir / "phoneme" / spk_id / f"{wav_id_path}.phone",
                    mel_path=self.data_dir / "mel" / spk_id / f"{wav_id_path}.npy",
                    xvector_path=self.data_dir / "xvector" / self.xvector_backend / spk_id / f"{wav_id_path}.npy",
                    phone_frames=frame_info[key],
                )
            )
        return records

    def _validate_contract(self, num_samples: int) -> None:
        sample_count = min(len(self.records), max(1, int(num_samples)))
        for idx in range(sample_count):
            record = self.records[idx]
            mel = np.load(record.mel_path)
            if mel.ndim != 2:
                raise ValueError(f"Mel must be rank-2, got shape {mel.shape} from {record.mel_path}")
            mel_dim = mel.shape[0] if mel.shape[0] == self.expected_mel_dim else mel.shape[-1]
            if mel_dim != self.expected_mel_dim:
                raise ValueError(
                    f"Mel dim mismatch for {record.mel_path}: expected {self.expected_mel_dim}, got {mel.shape}"
                )
            xvector = np.load(record.xvector_path)
            if xvector.ndim != 1 or xvector.shape[0] != self.expected_xvector_dim:
                raise ValueError(
                    f"Xvector dim mismatch for {record.xvector_path}: "
                    f"expected {self.expected_xvector_dim}, got {xvector.shape}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        phoneme_ids = torch.tensor(load_phoneme_ids(record.phone_path, self.phone_to_id), dtype=torch.long)
        mel = np.load(record.mel_path)
        if mel.shape[0] == self.expected_mel_dim:
            mel = torch.tensor(mel.T, dtype=torch.float32)
        elif mel.shape[1] == self.expected_mel_dim:
            mel = torch.tensor(mel, dtype=torch.float32)
        else:
            raise ValueError(f"Unexpected mel shape for {record.mel_path}: {mel.shape}")

        xvector = torch.tensor(np.load(record.xvector_path), dtype=torch.float32)
        if xvector.numel() != self.expected_xvector_dim:
            raise ValueError(
                f"xvector dim mismatch for {record.xvector_path}: "
                f"expected {self.expected_xvector_dim}, got {tuple(xvector.shape)}"
            )

        return {
            "wav_id": record.wav_id,
            "spk_id": record.spk_id,
            "phoneme_ids": phoneme_ids,
            "mel": mel,
            "xvector": xvector,
            "phoneme_len": int(phoneme_ids.numel()),
            "mel_len": int(mel.size(0)),
        }
