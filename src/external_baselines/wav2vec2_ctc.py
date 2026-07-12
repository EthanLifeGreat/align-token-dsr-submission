"""Pinned Wav2Vec2-CTC loading and audio helpers for the ASR--TTS baseline."""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from .common import COSYVOICE_SOURCE_REVISION, WAV2VEC2_MODEL_ID, WAV2VEC2_MODEL_REVISION, verify_git_revision


def resolve_checkpoint(cache_dir: Path | None = None, local_files_only: bool = False) -> Path:
    """Resolve the exact public Wav2Vec2 checkpoint without credentials."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=WAV2VEC2_MODEL_ID,
            revision=WAV2VEC2_MODEL_REVISION,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
        )
    )


def load_base_model(checkpoint_dir: Path, device: str):
    """Load the pinned legacy checkpoint with positional-convolution compatibility."""
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(str(checkpoint_dir), local_files_only=True)
    config = Wav2Vec2Config.from_pretrained(str(checkpoint_dir), local_files_only=True)
    model = Wav2Vec2ForCTC(config)
    state_path = checkpoint_dir / "pytorch_model.bin"
    if not state_path.is_file():
        model = Wav2Vec2ForCTC.from_pretrained(str(checkpoint_dir), local_files_only=True)
    else:
        state = torch.load(str(state_path), map_location="cpu")
        prefix = "wav2vec2.encoder.pos_conv_embed.conv."
        for old_key, new_key in {
            prefix + "weight_g": prefix + "parametrizations.weight.original0",
            prefix + "weight_v": prefix + "parametrizations.weight.original1",
        }.items():
            if old_key in state and new_key in model.state_dict():
                state[new_key] = state.pop(old_key)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Pinned Wav2Vec2 checkpoint did not load exactly: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
    model.to(device).eval()
    return processor, model


def load_finetuned_model(checkpoint_dir: Path, device: str):
    from transformers import AutoModelForCTC, AutoProcessor

    processor_dir = checkpoint_dir / "processor"
    model_dir = checkpoint_dir / "model"
    if not processor_dir.is_dir() or not model_dir.is_dir():
        raise FileNotFoundError("checkpoint must contain model/ and processor/")
    processor = AutoProcessor.from_pretrained(str(processor_dir), local_files_only=True)
    model = AutoModelForCTC.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    return processor, model


def load_audio_16k(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(path))
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if int(sample_rate) != 16000:
        audio = torchaudio.functional.resample(audio, int(sample_rate), 16000)
    return audio.squeeze(0).float()


@torch.inference_mode()
def transcribe_batch(processor, model, wav_paths: list[Path], device: str) -> list[str]:
    audio = [load_audio_16k(path).numpy() for path in wav_paths]
    inputs = processor(audio, sampling_rate=16000, padding=True, return_tensors="pt")
    model_inputs = {key: value.to(device) for key, value in inputs.items()}
    logits = model(**model_inputs).logits
    return [str(value).strip() for value in processor.batch_decode(logits.argmax(dim=-1).cpu())]


def load_cosyvoice2(cosyvoice_repo: Path, model_dir: Path, fp16: bool):
    """Load an externally installed, pinned CosyVoice checkout on demand."""
    import sys

    repo = cosyvoice_repo.resolve()
    if not repo.is_dir() or not model_dir.is_dir():
        raise FileNotFoundError("cosyvoice_repo and cosyvoice_model_dir must both exist")
    verify_git_revision(repo, COSYVOICE_SOURCE_REVISION, "CosyVoice source")
    sys.path.insert(0, str(repo))
    matcha = repo / "third_party" / "Matcha-TTS"
    if matcha.is_dir():
        sys.path.insert(0, str(matcha))
    from cosyvoice.cli.cosyvoice import CosyVoice2

    return CosyVoice2(str(model_dir.resolve()), load_jit=False, load_trt=False, fp16=fp16)


def synthesize_zero_shot(cosyvoice, text: str, prompt_text: str, prompt_wav: Path, output_wav: Path) -> None:
    prompt_speech = load_audio_16k(prompt_wav).unsqueeze(0)
    output = None
    for chunk in cosyvoice.inference_zero_shot(
        tts_text=text,
        prompt_text=prompt_text,
        prompt_speech_16k=prompt_speech,
        stream=False,
        speed=1.0,
        text_frontend=True,
    ):
        output = chunk
    if output is None or "tts_speech" not in output:
        raise RuntimeError("CosyVoice2 produced no tts_speech")
    waveform = output["tts_speech"].detach().cpu()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_wav), waveform, int(cosyvoice.sample_rate))
