"""Dataset for phoneme2token."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .frontend import load_audio_mono_16k


def canonical_wav_id(wav_id: object) -> str:
    value = str(wav_id).strip()
    if value.isdigit():
        return str(int(value))
    return value


def wav_id_candidates(wav_id: object) -> list[str]:
    value = str(wav_id).strip()
    candidates = [value]
    if value.isdigit():
        padded = f"{int(value):06d}"
        raw = str(int(value))
        for item in (padded, raw):
            if item not in candidates:
                candidates.append(item)
    return candidates


def load_phone_dict(path: Path) -> Dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as f:
        id_to_phone = json.load(f)
    return {phone: int(idx) for idx, phone in id_to_phone.items()}


def load_phoneme_names(path: Path) -> list[str]:
    names: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            names.append(line.split()[0])
    return names


def load_token_len_info(dataset_root: Path, token_dir_name: str) -> dict[tuple[str, str], dict]:
    token_len_csv = dataset_root / "token" / token_dir_name / "token_lens.csv"
    if not token_len_csv.exists():
        raise FileNotFoundError(f"token_lens.csv not found: {token_len_csv}")
    info: dict[tuple[str, str], dict] = {}
    with token_len_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_id = canonical_wav_id(row["wav_id"])
            spk_id = str(row["spk_id"]).strip()
            row["wav_id_path"] = str(row["wav_id"]).strip()
            info[(spk_id, wav_id)] = row
    return info


def resolve_xvector_path(data_root: Path, backend: str, spk_id: str, wav_id: str) -> Path:
    utterance_path = data_root / "xvector" / backend / spk_id / f"{wav_id}.npy"
    if utterance_path.exists():
        return utterance_path
    speaker_path = data_root / "xvector" / backend / f"{spk_id}.npy"
    if speaker_path.exists():
        return speaker_path
    raise FileNotFoundError(
        f"Missing xvector for spk={spk_id}, wav={wav_id}; tried "
        f"{utterance_path} and {speaker_path}"
    )


def build_wav_path(data_root: Path, spk_id: str, wav_id: str) -> Path:
    return data_root / "wav" / spk_id / f"{wav_id}.wav"


def build_phone_path(data_root: Path, spk_id: str, wav_id: str) -> Path:
    return data_root / "phoneme" / spk_id / f"{wav_id}.phone"


def build_token_path(data_root: Path, token_dir_name: str, spk_id: str, wav_id: str) -> Path:
    return data_root / "token" / token_dir_name / spk_id / f"{wav_id}.npy"


def build_xvector_path(data_root: Path, backend: str, spk_id: str, wav_id: str) -> Path:
    return data_root / "xvector" / backend / spk_id / f"{wav_id}.npy"


@dataclass(frozen=True)
class Phoneme2TokenRecord:
    wav_id: str
    spk_id: str
    split: str
    wav_path: Path
    phone_path: Path
    token_path: Path
    xvector_path: Path
    phoneme_50hz_len: int
    token_len: int
    base_len: int
    train_target_len: int

    @property
    def bucket_length(self) -> int:
        # xvector + phoneme prefix + BOS/target positions
        return 1 + self.phoneme_50hz_len + 1 + self.train_target_len


class Phoneme2TokenDataset(Dataset):
    """Loads 50Hz phoneme, xvector, and padded CosyVoice2 token targets."""

    def __init__(
        self,
        data_dir: str,
        dataset: str,
        split: str = "train",
        phone_dict: Optional[str] = None,
        valid_frames_csv: Optional[str] = None,
        token_dir_name: str = "s3tokenizer_v2_25hz",
        xvector_backend: str = "eresnet2v2",
        speech_vocab_size: int = 6561,
        token_pad_id: int = 6561,
        bos_id: int = 6562,
        length_pad_margin: int = 2,
        strict_files: bool = True,
        frontend_mode: str = "precomputed",
        max_samples: Optional[int] = None,
        wav2phoneme_ckpt: Optional[str] = None,
    ):
        self.data_root = Path(data_dir) / dataset
        self.dataset = dataset
        self.split = "dev" if split == "val" else split
        self.token_dir_name = token_dir_name
        self.xvector_backend = xvector_backend
        self.speech_vocab_size = int(speech_vocab_size)
        self.token_pad_id = int(token_pad_id)
        self.bos_id = int(bos_id)
        self.length_pad_margin = int(length_pad_margin)
        self.strict_files = bool(strict_files)
        self.frontend_mode = str(frontend_mode)
        self.max_samples = None if max_samples is None else int(max_samples)
        self.wav2phoneme_ckpt = None if wav2phoneme_ckpt is None else str(wav2phoneme_ckpt)

        phone_dict_path = Path(phone_dict) if phone_dict else self.data_root / "phone_dict.json"
        self.phone_to_id = load_phone_dict(phone_dict_path)
        self.sil_id = self.phone_to_id.get("sil", 0)
        self.num_phonemes = max(self.phone_to_id.values()) + 1
        self.token_len_info = load_token_len_info(self.data_root, self.token_dir_name)

        valid_info = self._load_valid_info(valid_frames_csv)
        meta = pd.read_csv(self.data_root / "meta.csv")
        if self.split in {"train", "dev", "test"}:
            meta = meta[meta["split"] == self.split]
        elif self.split != "all":
            raise ValueError(f"Unsupported split={split}")

        records: list[Phoneme2TokenRecord] = []
        skipped = 0
        for row in meta.itertuples(index=False):
            record = self._build_record(row, valid_info)
            if record is None:
                skipped += 1
                continue
            records.append(record)
            if self.max_samples is not None and len(records) >= self.max_samples:
                break

        if not records:
            raise ValueError(
                f"No usable samples for {dataset}/{self.split}; skipped={skipped}, "
                f"valid_frames_csv={valid_frames_csv}"
            )
        self.records = records
        self.lengths = [record.bucket_length for record in records]
        print(
            f"Loaded {len(records)} phoneme2token samples for {dataset}/{self.split}; "
            f"skipped={skipped}"
        )

    def _load_valid_info(self, valid_frames_csv: Optional[str]) -> Optional[dict[tuple[str, str], dict]]:
        if not valid_frames_csv:
            return None
        path = Path(valid_frames_csv)
        if not path.is_absolute():
            path = self.data_root / path
        if not path.exists():
            raise FileNotFoundError(f"valid_frames_csv not found: {path}")

        info: dict[tuple[str, str], dict] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "is_valid" in row and str(row["is_valid"]).strip() in {"0", "false", "False"}:
                    continue
                wav_id = canonical_wav_id(row["wav_id"])
                spk_id = str(row["spk_id"]).strip()
                info[(spk_id, wav_id)] = row
        return info

    def _build_record(self, row, valid_info: Optional[dict[tuple[str, str], dict]]):
        wav_id_raw = str(getattr(row, "wav_id")).strip()
        spk_id = str(getattr(row, "spk_id")).strip()
        split = str(getattr(row, "split")).strip()
        key = (spk_id, canonical_wav_id(wav_id_raw))
        valid_row = valid_info.get(key) if valid_info is not None else None
        if valid_info is not None and valid_row is None:
            return None

        token_row = self.token_len_info.get(key)
        if token_row is None:
            return None

        wav_id = str(token_row.get("wav_id_path") or wav_id_raw).strip()
        wav_path = build_wav_path(self.data_root, spk_id, wav_id)
        phone_path = build_phone_path(self.data_root, spk_id, wav_id)
        token_path = build_token_path(self.data_root, self.token_dir_name, spk_id, wav_id)
        xvector_path = build_xvector_path(self.data_root, self.xvector_backend, spk_id, wav_id)

        if valid_row is not None and "base_len" in valid_row and valid_row["base_len"]:
            base_len = int(valid_row["base_len"])
            token_len = int(valid_row.get("token_len") or token_row["token_len"])
            phoneme_50hz_len = int(
                valid_row.get("phoneme_50hz_len") or valid_row.get("phone_frames_ds2") or 0
            )
            if phoneme_50hz_len <= 0:
                phoneme_50hz_len = token_len * 2
            train_target_len = int(
                valid_row.get("train_target_len") or (base_len + self.length_pad_margin)
            )
        else:
            token_len = int(token_row["token_len"])
            if self.frontend_mode == "wav2phoneme_ctc_frameargmax":
                phoneme_50hz_len = token_len * 2
            else:
                if not phone_path.exists():
                    if self.strict_files:
                        raise FileNotFoundError(f"Missing phoneme file: {phone_path}")
                    return None
                phoneme_len = len(load_phoneme_names(phone_path))
                phoneme_50hz_len = (phoneme_len + 1) // 2
            base_len = round(phoneme_50hz_len / 2)
            train_target_len = base_len + self.length_pad_margin
            if token_len > train_target_len:
                return None

        return Phoneme2TokenRecord(
            wav_id=wav_id,
            spk_id=spk_id,
            split=split,
            wav_path=wav_path,
            phone_path=phone_path,
            token_path=token_path,
            xvector_path=xvector_path,
            phoneme_50hz_len=phoneme_50hz_len,
            token_len=token_len,
            base_len=base_len,
            train_target_len=train_target_len,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        tokens = np.load(record.token_path).reshape(-1).astype(np.int64)
        if tokens.size != record.token_len:
            raise ValueError(f"Token length changed for {record.token_path}")
        if tokens.size and (tokens.min() < 0 or tokens.max() >= self.speech_vocab_size):
            raise ValueError(
                f"Token ids out of range [0, {self.speech_vocab_size - 1}] in {record.token_path}"
            )
        if tokens.size > record.train_target_len:
            raise ValueError(
                f"token_len={tokens.size} exceeds train_target_len={record.train_target_len} "
                f"for {record.spk_id}/{record.wav_id}"
            )

        labels = torch.full((record.train_target_len,), self.token_pad_id, dtype=torch.long)
        if tokens.size:
            labels[: tokens.size] = torch.from_numpy(tokens)
        decoder_input_tokens = torch.empty_like(labels)
        decoder_input_tokens[0] = self.bos_id
        if labels.numel() > 1:
            decoder_input_tokens[1:] = labels[:-1]

        xvector = torch.from_numpy(np.load(record.xvector_path).astype(np.float32))

        if self.frontend_mode == "wav2phoneme_ctc_frameargmax":
            waveform = load_audio_mono_16k(record.wav_path)
            return {
                "wav_id": record.wav_id,
                "spk_id": record.spk_id,
                "dataset_name": self.dataset,
                "waveform": waveform,
                "wav_path": str(record.wav_path),
                "decoder_input_tokens": decoder_input_tokens,
                "labels": labels,
                "xvector": xvector,
                "phoneme_len": record.phoneme_50hz_len,
                "target_len": labels.numel(),
                "token_len": record.token_len,
                "base_len": record.base_len,
            }

        names = load_phoneme_names(record.phone_path)
        phoneme_ids = [self.phone_to_id.get(name, self.sil_id) for name in names]
        phoneme_ids = torch.tensor(phoneme_ids[0::2], dtype=torch.long)

        return {
            "wav_id": record.wav_id,
            "spk_id": record.spk_id,
            "dataset_name": self.dataset,
            "phoneme_ids": phoneme_ids,
            "decoder_input_tokens": decoder_input_tokens,
            "labels": labels,
            "xvector": xvector,
            "phoneme_len": phoneme_ids.numel(),
            "target_len": labels.numel(),
            "token_len": record.token_len,
            "base_len": record.base_len,
        }
