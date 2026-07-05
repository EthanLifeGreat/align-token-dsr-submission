from __future__ import annotations

import base64
import io
import logging
import os
import random
import re
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from jiwer import wer as jiwer_wer
from pypinyin import Style, lazy_pinyin
from scipy.signal import resample

from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice, CosyVoice2, CosyVoice3

try:
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
except Exception:  # pragma: no cover - optional dependency
    ZhNormalizer = None

try:
    from omnisense.models import OmniSenseVoiceSmall
except Exception:  # pragma: no cover - optional dependency
    OmniSenseVoiceSmall = None

try:
    from funasr import AutoModel as FunASRAutoModel
except Exception:  # pragma: no cover - optional dependency
    FunASRAutoModel = None

try:
    from .schemas import Token2RewardsRequest, Token2WavRequest, Token2WerRequest
except ImportError:  # pragma: no cover
    from schemas import Token2RewardsRequest, Token2WavRequest, Token2WerRequest


logger = logging.getLogger(__name__)

ORIGINAL_VOCAB_SIZE = 151663
ASR_SAMPLE_RATE = 16000
DEFAULT_FRAME_RATE = 50


class _FunASRSenseVoiceWrapper:
    def __init__(self, model_id: str, device: str):
        self.model = FunASRAutoModel(
            model=model_id,
            device=device,
            disable_update=True,
        )

    def transcribe_single_batch(
        self,
        audios: list[np.ndarray],
        *,
        language: str,
        textnorm: str,
    ) -> list[Any]:
        use_itn = textnorm != "woitn"
        results = self.model.generate(
            input=audios,
            cache={},
            language=language,
            use_itn=use_itn,
            batch_size=1,
        )
        wrapped_results = []
        for item in results:
            wrapped_results.append(type("FunASRResult", (), {"text": str(item.get("text", ""))})())
        return wrapped_results


@dataclass
class ServiceConfig:
    cosyvoice_model_dir: str
    asr_model_id: str = "iic/SenseVoiceSmall"
    spk_embed_model_id: str = ""
    dnsmos_model_id: str = ""
    prompt_dataset_id: str = "yuekai/aishell"
    prompt_dataset_split: str = "test"
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda:0"
    fp16: bool = True
    max_audio_seconds: float = 30.0
    seed: int = 42


class TokenService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.loaded_at: datetime | None = None
        self.cosyvoice = None
        self.sample_rate = 0
        self.asr_model = None
        self.prompt_dataset = None
        self.device = "cpu"
        self.random = random.Random(config.seed)
        self.decode_lock = Lock()
        self.asr_lock = Lock()
        self.prompt_dataset_lock = Lock()
        self.zh_normalizer = (
            ZhNormalizer(
                cache_dir="./cache",
                remove_erhua=False,
                remove_interjections=False,
                remove_puncts=True,
                overwrite_cache=True,
            )
            if ZhNormalizer is not None
            else None
        )

    def load(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.device = self._resolve_device(self.config.device)
        if self.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(self.device))

        logger.info("Loading CosyVoice model from %s on %s", self.config.cosyvoice_model_dir, self.device)
        self.cosyvoice = self._load_cosyvoice_model(self.config.cosyvoice_model_dir)
        self.sample_rate = int(self.cosyvoice.sample_rate)
        self.loaded_at = datetime.now(timezone.utc)
        logger.info(
            "Token service ready: sample_rate=%s, prompt_dataset/asr deferred until first use",
            self.sample_rate,
        )

    def healthz(self) -> dict[str, Any]:
        if self.loaded_at is None:
            raise RuntimeError("Service has not finished loading.")
        return {
            "status": "ok",
            "cosyvoice_model_dir": self.config.cosyvoice_model_dir,
            "asr_model_id": self.config.asr_model_id,
            "device": self.device,
            "loaded_at": self.loaded_at.isoformat(),
            "token2rewards_is_mock": True,
        }

    def token2wav(self, request: Token2WavRequest) -> bytes:
        tokens = self._prepare_token_tensor(request.tokens, "tokens")
        prompt_token, prompt_feat, embedding = self._build_decode_condition(request, allow_random=True)
        audio = self._decode_audio(tokens, prompt_token, prompt_feat, embedding, request.speed)
        return self._encode_wav(audio, self.sample_rate)

    def token2wer(self, request: Token2WerRequest) -> dict[str, Any]:
        tokens = self._prepare_token_tensor(request.tokens, "tokens")
        prompt_token, prompt_feat, embedding = self._build_decode_condition(request, allow_random=True)
        audio = self._decode_audio(tokens, prompt_token, prompt_feat, embedding, speed=1.0)
        asr_hyp_text = self._run_asr(audio)
        ground_truth_text_norm = self._normalize_text(request.ground_truth_text)
        asr_hyp_text_norm = self._normalize_text(asr_hyp_text)
        return {
            "wer": self._compute_wer(ground_truth_text_norm, asr_hyp_text_norm),
            "asr_hyp_text": asr_hyp_text,
            "ground_truth_text_norm": ground_truth_text_norm,
            "asr_hyp_text_norm": asr_hyp_text_norm,
        }

    def token2rewards(self, request: Token2RewardsRequest) -> dict[str, Any]:
        self._prepare_token_tensor(request.tokens, "tokens")
        self._prepare_float_vector(request.spk_xvector_centroid, "spk_xvector_centroid")
        self._build_decode_condition(request, allow_random=False)
        return {
            "r_asr": 0.0,
            "r_spk_sim": 0.0,
            "r_dns_mos": 0.0,
        }

    def _load_cosyvoice_model(self, model_dir: str):
        if not os.path.exists(model_dir):
            return AutoModel(
                model_dir=model_dir,
                load_jit=True,
                load_trt=True,
                fp16=self.config.fp16,
            )

        for config_name, model_cls in (
            ("cosyvoice3.yaml", CosyVoice3),
            ("cosyvoice2.yaml", CosyVoice2),
            ("cosyvoice.yaml", CosyVoice),
        ):
            if os.path.exists(os.path.join(model_dir, config_name)):
                return model_cls(
                    model_dir=model_dir,
                    load_jit=True,
                    load_trt=True,
                    fp16=self.config.fp16,
                )

        return AutoModel(
            model_dir=model_dir,
            load_jit=True,
            load_trt=True,
            fp16=self.config.fp16,
        )

    def _resolve_device(self, requested: str) -> str:
        if requested.startswith("cuda") and torch.cuda.is_available():
            return requested
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to cpu")
        return "cpu"

    def _load_asr_model(self, model_id: str, device: str):
        if OmniSenseVoiceSmall is not None:
            device_id = 0
            if device.startswith("cuda"):
                parts = device.split(":")
                if len(parts) == 2 and parts[1].isdigit():
                    device_id = int(parts[1])
            return OmniSenseVoiceSmall(model_id, quantize=False, device_id=device_id)
        if FunASRAutoModel is not None:
            logger.warning("`omnisense` is unavailable, falling back to FunASR AutoModel for /token2wer")
            return _FunASRSenseVoiceWrapper(model_id, device=device)
        raise RuntimeError(
            "`omnisense` and `funasr` are both unavailable. Install PytritonSenseVoice or FunASR to enable /token2wer."
        )

    def _load_prompt_dataset(self, dataset_id: str, split_name: str):
        load_attempts = (
            {"path": dataset_id, "split": split_name, "trust_remote_code": True},
            {
                "path": dataset_id,
                "name": split_name,
                "split": split_name,
                "trust_remote_code": True,
            },
        )
        last_error = None
        for kwargs in load_attempts:
            try:
                dataset = load_dataset(**kwargs)
                return self._coerce_prompt_dataset(dataset, split_name)
            except Exception as exc:  # pragma: no cover - depends on remote dataset shape
                last_error = exc
        try:
            dataset = load_dataset(dataset_id, split_name, trust_remote_code=True)
            return self._coerce_prompt_dataset(dataset, split_name)
        except Exception as exc:  # pragma: no cover - depends on remote dataset shape
            last_error = exc
        raise RuntimeError(
            f"Failed to load prompt dataset {dataset_id!r} with split/config {split_name!r}: {last_error}"
        )

    def _coerce_prompt_dataset(self, dataset: Dataset | DatasetDict, split_name: str) -> Dataset:
        if isinstance(dataset, Dataset):
            return dataset
        if isinstance(dataset, DatasetDict):
            if split_name in dataset:
                return dataset[split_name]
            available_splits = ", ".join(dataset.keys())
            raise RuntimeError(
                f"Prompt dataset split {split_name!r} not found. Available splits: {available_splits}"
            )
        raise RuntimeError(f"Unsupported prompt dataset type: {type(dataset)!r}")

    def _build_decode_condition(self, request: Any, allow_random: bool):
        prompt_wav_base64 = getattr(request, "prompt_wav_base64", None)
        prompt_wav_path = getattr(request, "prompt_wav_path", None)
        if prompt_wav_path is not None:
            return self._explicit_prompt_condition_from_path(prompt_wav_path)
        if prompt_wav_base64 is None:
            if not allow_random:
                raise ValueError("This endpoint requires `prompt_wav_base64`.")
            return self._random_prompt_condition()
        return self._explicit_prompt_condition(prompt_wav_base64)

    def _random_prompt_condition(self):
        prompt_dataset = self._ensure_prompt_dataset_loaded()
        sample = prompt_dataset[self.random.randrange(len(prompt_dataset))]
        audio_data = sample["audio"]
        prompt_text = sample["text"].replace(" ", "")
        prompt_audio = np.asarray(audio_data["array"], dtype=np.float32)
        prompt_sr = int(audio_data["sampling_rate"])
        if prompt_sr != ASR_SAMPLE_RATE:
            prompt_audio = resample(prompt_audio, int(round(len(prompt_audio) * ASR_SAMPLE_RATE / prompt_sr)))
        prompt_speech_16k = torch.from_numpy(prompt_audio).float().unsqueeze(0)
        model_inputs = self.cosyvoice.frontend.frontend_zero_shot(
            "",
            prompt_text,
            prompt_speech_16k,
            self.sample_rate,
            "",
        )
        return (
            model_inputs["flow_prompt_speech_token"].detach().cpu().to(torch.int32),
            model_inputs["prompt_speech_feat"].detach().cpu(),
            model_inputs["flow_embedding"].detach().cpu(),
        )

    def _explicit_prompt_condition(self, prompt_wav_base64: str):
        wav_bytes = self._decode_prompt_wav_bytes(prompt_wav_base64)
        return self._explicit_prompt_condition_from_bytes(wav_bytes)

    def _explicit_prompt_condition_from_path(self, prompt_wav_path: str):
        wav_path = Path(prompt_wav_path)
        if not wav_path.is_file():
            raise ValueError(f"`prompt_wav_path` does not exist: {prompt_wav_path}")
        try:
            wav_bytes = wav_path.read_bytes()
        except Exception as exc:
            raise ValueError(f"Failed to read `prompt_wav_path`: {prompt_wav_path}") from exc
        return self._explicit_prompt_condition_from_bytes(wav_bytes)

    def _explicit_prompt_condition_from_bytes(self, wav_bytes: bytes):
        frontend = self.cosyvoice.frontend
        with self.decode_lock, torch.inference_mode():
            prompt_token, _ = frontend._extract_speech_token(io.BytesIO(wav_bytes))
            prompt_feat, _ = frontend._extract_speech_feat(io.BytesIO(wav_bytes))
            embedding = frontend._extract_spk_embedding(io.BytesIO(wav_bytes))
        return (
            prompt_token.detach().cpu().to(torch.int32),
            prompt_feat.detach().cpu(),
            embedding.detach().cpu(),
        )

    def _decode_prompt_wav_bytes(self, prompt_wav_base64: str) -> bytes:
        try:
            wav_bytes = base64.b64decode(prompt_wav_base64, validate=True)
        except Exception as exc:
            raise ValueError("`prompt_wav_base64` is not valid base64.") from exc
        if not wav_bytes:
            raise ValueError("`prompt_wav_base64` decoded to empty audio bytes.")
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                if wav_file.getnframes() <= 0:
                    raise ValueError("`prompt_wav_base64` contains an empty WAV file.")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("`prompt_wav_base64` must encode a valid WAV file.") from exc
        return wav_bytes

    def _prepare_token_tensor(self, values: list[int], field_name: str) -> torch.Tensor:
        array = np.asarray(values, dtype=np.int64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"`{field_name}` must be a non-empty 1D integer list.")
        if np.isnan(array.astype(np.float64)).any():
            raise ValueError(f"`{field_name}` must not contain NaN.")
        max_token_length = int(self.config.max_audio_seconds * self._input_frame_rate())
        if array.size > max_token_length:
            raise ValueError(
                f"`{field_name}` length {array.size} exceeds max token length {max_token_length} "
                f"derived from max_audio_seconds={self.config.max_audio_seconds}."
            )
        if array.max(initial=0) >= ORIGINAL_VOCAB_SIZE:
            array = array - ORIGINAL_VOCAB_SIZE
        if array.min(initial=0) < 0:
            raise ValueError(f"`{field_name}` contains negative token ids after normalization.")
        return torch.from_numpy(array.astype(np.int32)).unsqueeze(0)

    def _prepare_float_vector(self, values: list[float], field_name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"`{field_name}` must be a non-empty 1D float list.")
        if not np.isfinite(array).all():
            raise ValueError(f"`{field_name}` must only contain finite values.")
        return array

    def _input_frame_rate(self) -> int:
        return int(getattr(self.cosyvoice.model.flow, "input_frame_rate", DEFAULT_FRAME_RATE))

    def _decode_audio(
        self,
        tokens: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        speed: float,
    ) -> np.ndarray:
        with self.decode_lock, torch.inference_mode():
            model_output = self.cosyvoice.model.tts(
                source_speech_token=tokens,
                flow_prompt_speech_token=prompt_token,
                prompt_speech_feat=prompt_feat,
                flow_embedding=embedding,
                stream=False,
                speed=speed,
            )
            try:
                output = next(model_output)
            except StopIteration as exc:
                raise RuntimeError("CosyVoice decode returned no audio.") from exc
        audio = output["tts_speech"].squeeze(0).detach().float().cpu().numpy()
        if audio.size == 0:
            raise RuntimeError("Decoded audio is empty.")
        if not np.isfinite(audio).all():
            raise RuntimeError("Decoded audio contains NaN or Inf.")
        return audio

    def _run_asr(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        asr_model = self._ensure_asr_loaded()
        if self.sample_rate != ASR_SAMPLE_RATE:
            num_samples = int(round(len(audio) * ASR_SAMPLE_RATE / self.sample_rate))
            if num_samples <= 0:
                return ""
            audio = resample(audio, num_samples)
        with self.asr_lock:
            results = asr_model.transcribe_single_batch(
                [audio.astype(np.float32)],
                language="zh",
                textnorm="woitn",
            )
        if not results:
            return ""
        return str(results[0].text)

    def _ensure_prompt_dataset_loaded(self):
        if self.prompt_dataset is not None:
            return self.prompt_dataset
        with self.prompt_dataset_lock:
            if self.prompt_dataset is None:
                logger.info(
                    "Loading prompt dataset %s split/config %s",
                    self.config.prompt_dataset_id,
                    self.config.prompt_dataset_split,
                )
                self.prompt_dataset = self._load_prompt_dataset(
                    self.config.prompt_dataset_id,
                    self.config.prompt_dataset_split,
                )
        return self.prompt_dataset

    def _ensure_asr_loaded(self):
        if self.asr_model is not None:
            return self.asr_model
        with self.asr_lock:
            if self.asr_model is None:
                logger.info("Loading ASR model %s", self.config.asr_model_id)
                self.asr_model = self._load_asr_model(self.config.asr_model_id, self.device)
        return self.asr_model

    def _normalize_text(self, text: str) -> str:
        text = text or ""
        text = re.sub(r"<\|[^|]+\|>", "", text)
        if self.zh_normalizer is not None:
            text = self.zh_normalizer.normalize(text)
        text = text.lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        return text

    def _compute_wer(self, reference: str, hypothesis: str) -> float:
        if not reference and not hypothesis:
            return 0.0
        if not reference and hypothesis:
            return 1.0
        ref_pinyin = lazy_pinyin(
            reference,
            style=Style.TONE3,
            tone_sandhi=True,
            neutral_tone_with_five=True,
        )
        hyp_pinyin = lazy_pinyin(
            hypothesis,
            style=Style.TONE3,
            tone_sandhi=True,
            neutral_tone_with_five=True,
        )
        return float(jiwer_wer(" ".join(ref_pinyin), " ".join(hyp_pinyin)))

    def _encode_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return buffer.getvalue()
