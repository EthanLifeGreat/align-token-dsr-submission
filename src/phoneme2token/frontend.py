"""On-the-fly wav2phoneme CTC frameargmax frontend helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from src.wav2phoneme.model import Wav2PhonemeModel


def load_audio_mono_16k(audio_path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
    audio_path = Path(audio_path)
    candidate_paths = [audio_path]
    if audio_path.suffix == ".wav":
        candidate_paths.append(audio_path.with_suffix(".WAV"))
    elif audio_path.suffix == ".WAV":
        candidate_paths.append(audio_path.with_suffix(".wav"))

    last_error: Optional[Exception] = None
    for candidate in candidate_paths:
        try:
            waveform, sr = torchaudio.load(str(candidate))
            break
        except (FileNotFoundError, RuntimeError) as exc:
            last_error = exc
    else:
        raise FileNotFoundError(f"Unable to load audio from {audio_path}") from last_error

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).contiguous()


def pad_audio_batch(waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not waveforms:
        raise ValueError("waveforms must not be empty")
    batch = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([int(item.numel()) for item in waveforms], dtype=torch.long)
    steps = torch.arange(int(batch.size(1)), dtype=torch.long).unsqueeze(0)
    attention_mask = steps.lt(lengths.unsqueeze(1)).to(dtype=torch.long)
    return batch, attention_mask


def repeat_upsample_ids(
    ids_50hz: list[int],
    *,
    repeat_factor: int = 2,
) -> tuple[list[int], list[int], bool]:
    if repeat_factor <= 0:
        raise ValueError("repeat_factor must be positive")
    ids_100hz: list[int] = []
    for item in ids_50hz:
        ids_100hz.extend([int(item)] * repeat_factor)
    restored = ids_100hz[0::repeat_factor]
    return ids_100hz, restored, restored == ids_50hz


@lru_cache(maxsize=8)
def _load_length_estimator_model(checkpoint_path: str) -> Wav2PhonemeModel:
    model = Wav2PhonemeModel.from_pretrained(checkpoint_path)
    model.eval()
    return model


def estimate_phoneme_50hz_length(
    *,
    checkpoint_path: str,
    num_samples: int,
) -> int:
    model = _load_length_estimator_model(str(checkpoint_path))
    input_lengths = model.wav2vec._get_feat_extract_output_lengths(
        torch.tensor([int(num_samples)], dtype=torch.long)
    )
    return int(input_lengths.tolist()[0])


@dataclass
class FrontendSample:
    ids_50hz: list[int]
    ids_100hz: list[int]
    restored_ids_50hz: list[int]
    blank_ratio: float
    nonblank_frames_50hz: int
    nonblank_frames_restored: int
    identity_check_pass: bool


@dataclass
class FrontendBatch:
    phoneme_ids: torch.Tensor
    phoneme_key_padding_mask: torch.Tensor
    metrics: dict[str, float]
    all_identity_pass: bool
    identity_pass_flags: list[bool]


class FrameArgmaxFrontend:
    def __init__(
        self,
        wav2phoneme_ckpt: str,
        phone_dict_path: str,
        *,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        repeat_factor: int = 2,
        blank_id: Optional[int] = None,
    ):
        self.wav2phoneme_ckpt = str(wav2phoneme_ckpt)
        self.phone_dict_path = str(phone_dict_path)
        self.device = str(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.repeat_factor = int(repeat_factor)
        self.blank_id = None if blank_id is None else int(blank_id)
        self.__dict__["_model"] = None

    @property
    def model(self) -> Wav2PhonemeModel:
        model = self.__dict__.get("_model")
        if model is None:
            model = Wav2PhonemeModel.from_pretrained(self.wav2phoneme_ckpt)
            model.to(self.device)
            model.eval()
            model.requires_grad_(False)
            self.__dict__["_model"] = model
            if self.blank_id is None:
                self.blank_id = int(getattr(model.config, "ctc_blank_id", 0))
        return model

    def cache_file(self, dataset_name: str, spk_id: str, wav_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / dataset_name / spk_id / f"{wav_id}.json"

    def load_cached(self, dataset_name: str, spk_id: str, wav_id: str) -> Optional[FrontendSample]:
        if self.cache_dir is None:
            return None
        cache_file = self.cache_file(dataset_name, spk_id, wav_id)
        if cache_file is None or not cache_file.exists():
            return None
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return FrontendSample(
            ids_50hz=[int(item) for item in payload["ids_50hz"]],
            ids_100hz=[int(item) for item in payload["ids_100hz"]],
            restored_ids_50hz=[int(item) for item in payload["restored_ids_50hz"]],
            blank_ratio=float(payload["blank_ratio"]),
            nonblank_frames_50hz=int(payload["nonblank_frames_50hz"]),
            nonblank_frames_restored=int(payload["nonblank_frames_restored"]),
            identity_check_pass=bool(payload["identity_check_pass"]),
        )

    def save_cached(
        self,
        dataset_name: str,
        spk_id: str,
        wav_id: str,
        sample: FrontendSample,
    ) -> None:
        if self.cache_dir is None:
            return
        cache_file = self.cache_file(dataset_name, spk_id, wav_id)
        if cache_file is None:
            return
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids_50hz": sample.ids_50hz,
            "ids_100hz": sample.ids_100hz,
            "restored_ids_50hz": sample.restored_ids_50hz,
            "blank_ratio": sample.blank_ratio,
            "nonblank_frames_50hz": sample.nonblank_frames_50hz,
            "nonblank_frames_restored": sample.nonblank_frames_restored,
            "identity_check_pass": sample.identity_check_pass,
            "repeat_factor": self.repeat_factor,
            "blank_id": self.blank_id,
            "wav2phoneme_ckpt": self.wav2phoneme_ckpt,
            "phone_dict_path": self.phone_dict_path,
        }
        cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _build_sample(self, ids_50hz: list[int]) -> FrontendSample:
        ids_100hz, restored_ids_50hz, identity_check_pass = repeat_upsample_ids(
            ids_50hz,
            repeat_factor=self.repeat_factor,
        )
        nonblank_frames_50hz = sum(int(item != self.blank_id) for item in ids_50hz)
        nonblank_frames_restored = sum(int(item != self.blank_id) for item in restored_ids_50hz)
        blank_ratio = 0.0
        if ids_50hz:
            blank_ratio = 1.0 - (nonblank_frames_50hz / float(len(ids_50hz)))
        return FrontendSample(
            ids_50hz=ids_50hz,
            ids_100hz=ids_100hz,
            restored_ids_50hz=restored_ids_50hz,
            blank_ratio=blank_ratio,
            nonblank_frames_50hz=nonblank_frames_50hz,
            nonblank_frames_restored=nonblank_frames_restored,
            identity_check_pass=identity_check_pass,
        )

    @torch.no_grad()
    def extract_file(
        self,
        *,
        audio_path: str | Path,
        dataset_name: str,
        spk_id: str,
        wav_id: str,
    ) -> FrontendSample:
        cached = self.load_cached(dataset_name, spk_id, wav_id)
        if cached is not None:
            return cached
        waveform = load_audio_mono_16k(audio_path)
        input_values, attention_mask = pad_audio_batch([waveform])
        sample = self.extract_batch(
            input_values=input_values,
            attention_mask=attention_mask,
            dataset_names=[dataset_name],
            spk_ids=[spk_id],
            wav_ids=[wav_id],
        )[0]
        return sample

    @torch.no_grad()
    def extract_batch(
        self,
        *,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        dataset_names: list[str],
        spk_ids: list[str],
        wav_ids: list[str],
    ) -> list[FrontendSample]:
        if not (
            len(dataset_names)
            == len(spk_ids)
            == len(wav_ids)
            == int(input_values.size(0))
            == int(attention_mask.size(0))
        ):
            raise ValueError("Batch metadata length mismatch in FrameArgmaxFrontend.extract_batch")

        results: list[Optional[FrontendSample]] = [None] * int(input_values.size(0))
        miss_indices: list[int] = []
        for index, (dataset_name, spk_id, wav_id) in enumerate(zip(dataset_names, spk_ids, wav_ids)):
            cached = self.load_cached(dataset_name, spk_id, wav_id)
            if cached is not None:
                results[index] = cached
            else:
                miss_indices.append(index)

        if miss_indices:
            miss_tensor = torch.tensor(miss_indices, dtype=torch.long, device=input_values.device)
            miss_input = input_values.index_select(0, miss_tensor).to(self.device)
            miss_mask = attention_mask.index_select(0, miss_tensor).to(self.device)
            logits = self.model(
                input_values=miss_input,
                attention_mask=miss_mask,
                return_loss=False,
            )
            lengths = self.model._get_encoder_output_lengths(miss_mask, logits.shape[1])
            pred = logits.argmax(dim=-1).detach().cpu()
            lengths = lengths.detach().cpu().tolist()
            for local_index, batch_index in enumerate(miss_indices):
                ids_50hz = pred[local_index, : int(lengths[local_index])].tolist()
                sample = self._build_sample([int(item) for item in ids_50hz])
                results[batch_index] = sample
                self.save_cached(dataset_names[batch_index], spk_ids[batch_index], wav_ids[batch_index], sample)

        finalized = [item for item in results if item is not None]
        if len(finalized) != len(results):
            raise RuntimeError("Frontend extraction returned incomplete batch")
        return finalized

    @torch.no_grad()
    def extract_batch_tensors(
        self,
        *,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        dataset_names: list[str],
        spk_ids: list[str],
        wav_ids: list[str],
        pad_id: int,
        output_device: torch.device,
    ) -> FrontendBatch:
        samples = self.extract_batch(
            input_values=input_values,
            attention_mask=attention_mask,
            dataset_names=dataset_names,
            spk_ids=spk_ids,
            wav_ids=wav_ids,
        )
        phoneme_sequences = [
            torch.tensor(sample.restored_ids_50hz, dtype=torch.long) for sample in samples
        ]
        phoneme_ids = pad_sequence(
            phoneme_sequences,
            batch_first=True,
            padding_value=int(pad_id),
        ).to(output_device)
        phoneme_key_padding_mask = phoneme_ids.eq(int(pad_id))
        metrics = {
            "frontend_blank_ratio": 0.0,
            "frontend_nonblank_frames": 0.0,
            "frontend_nonblank_frames_restored": 0.0,
            "frontend_phone_frames_100hz": 0.0,
            "frontend_identity_pass_rate": 0.0,
        }
        all_identity_pass = True
        for sample in samples:
            metrics["frontend_blank_ratio"] += sample.blank_ratio
            metrics["frontend_nonblank_frames"] += float(sample.nonblank_frames_50hz)
            metrics["frontend_nonblank_frames_restored"] += float(
                sample.nonblank_frames_restored
            )
            metrics["frontend_phone_frames_100hz"] += float(len(sample.ids_100hz))
            metrics["frontend_identity_pass_rate"] += float(sample.identity_check_pass)
            all_identity_pass = all_identity_pass and bool(sample.identity_check_pass)
        batch_size = float(len(samples))
        for key in metrics:
            metrics[key] /= batch_size
        return FrontendBatch(
            phoneme_ids=phoneme_ids,
            phoneme_key_padding_mask=phoneme_key_padding_mask,
            metrics=metrics,
            all_identity_pass=all_identity_pass,
            identity_pass_flags=[bool(sample.identity_check_pass) for sample in samples],
        )
