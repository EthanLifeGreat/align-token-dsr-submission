#!/usr/bin/env python
"""Run validation-selected Wav2Vec2-CTC ASR followed by CosyVoice2 C0 synthesis."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.external_baselines.common import (
    COSYVOICE_MODEL_ID,
    COSYVOICE_MODEL_REVISION,
    COSYVOICE_SOURCE_REVISION,
    WAV2VEC2_MODEL_ID,
    WAV2VEC2_MODEL_REVISION,
    append_csv,
    load_manifest,
    normalize_asr_text,
    read_csv,
    truncate_error,
    wav_duration,
    write_json,
    write_silence,
)
from src.external_baselines.wav2vec2_ctc import (
    load_cosyvoice2,
    load_finetuned_model,
    synthesize_zero_shot,
    transcribe_batch,
)


ASR_FIELDS = [
    "utt_id", "speaker_id", "source_wav", "reference_text", "asr_raw_text", "asr_norm_text", "status", "error"
]
MANIFEST_FIELDS = [
    "utt_id", "speaker_id", "source_wav", "reference_text", "asr_raw_text", "asr_norm_text",
    "generated_wav", "prompt_wav", "prompt_text", "condition", "status", "error",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.is_file() and path.stat().st_size else []


def require_new_or_resumable(path: Path, resume: bool) -> None:
    if path.exists() and not resume:
        raise FileExistsError(f"Refusing to append to existing {path}; use --resume or a new output directory")


def read_prompt_text(args: argparse.Namespace) -> str:
    if args.prompt_text_file is not None:
        text = args.prompt_text_file.read_text(encoding="utf-8").strip()
    else:
        text = str(args.prompt_text or "").strip()
    if not text:
        raise ValueError("provide a non-empty --prompt-text-file or --prompt-text")
    return text


def asr_row(item, raw_text: str = "", error: str = "") -> dict[str, str]:
    normalized = normalize_asr_text(raw_text)
    status = "success" if normalized else ("asr_failed" if error else "empty_asr")
    return {
        "utt_id": item.utt_id,
        "speaker_id": item.speaker_id,
        "source_wav": str(item.source_wav),
        "reference_text": item.reference_text,
        "asr_raw_text": raw_text,
        "asr_norm_text": normalized,
        "status": status,
        "error": error,
    }


def run_asr(args: argparse.Namespace, items) -> None:
    asr_manifest = args.output_dir / "asr_manifest.csv"
    require_new_or_resumable(asr_manifest, args.resume)
    completed = {row["utt_id"] for row in read_rows(asr_manifest)} if args.resume else set()
    pending = [item for item in items if item.utt_id not in completed]
    if not pending:
        return
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for ASR decoding")
    checkpoint = args.checkpoint.resolve()
    log_path = args.output_dir / "logs" / "wav2vec2_asr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        processor, model = load_finetuned_model(checkpoint, args.device)
        for start in tqdm(range(0, len(pending), args.asr_batch_size), desc="wav2vec2-ctc"):
            batch = pending[start : start + args.asr_batch_size]
            try:
                raw_texts = transcribe_batch(processor, model, [item.source_wav for item in batch], args.device)
                for item, raw_text in zip(batch, raw_texts, strict=True):
                    append_csv(asr_manifest, asr_row(item, raw_text), ASR_FIELDS)
            except Exception as batch_error:  # noqa: BLE001
                print(f"Batch failed; retrying individually: {batch_error!r}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for item in batch:
                    try:
                        append_csv(
                            asr_manifest,
                            asr_row(item, transcribe_batch(processor, model, [item.source_wav], args.device)[0]),
                            ASR_FIELDS,
                        )
                    except Exception as error:  # noqa: BLE001
                        print(traceback.format_exc(), flush=True)
                        append_csv(asr_manifest, asr_row(item, error=truncate_error(f"{type(error).__name__}: {error}")), ASR_FIELDS)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_tts(args: argparse.Namespace, items, prompt_text: str) -> None:
    asr_manifest = args.output_dir / "asr_manifest.csv"
    if not asr_manifest.is_file():
        raise FileNotFoundError("ASR output is missing; run --stage asr first")
    asr_rows = {row["utt_id"]: row for row in read_rows(asr_manifest)}
    expected = {item.utt_id for item in items}
    if set(asr_rows) != expected:
        raise ValueError(f"ASR/test utterance mismatch: missing={len(expected-set(asr_rows))}, extra={len(set(asr_rows)-expected)}")
    manifest = args.output_dir / "manifest.csv"
    require_new_or_resumable(manifest, args.resume)
    completed = {row["utt_id"] for row in read_rows(manifest)} if args.resume else set()
    pending = [item for item in items if item.utt_id not in completed]
    if not pending:
        return
    if args.cosyvoice_repo is None or args.cosyvoice_model_dir is None or args.prompt_wav is None:
        raise ValueError("TTS requires --cosyvoice-repo, --cosyvoice-model-dir, and --prompt-wav")
    prompt_wav = args.prompt_wav.resolve()
    if not prompt_wav.is_file():
        raise FileNotFoundError(prompt_wav)
    log_path = args.output_dir / "logs" / "cosyvoice2_generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        cosyvoice = load_cosyvoice2(args.cosyvoice_repo, args.cosyvoice_model_dir, args.fp16)
        for item in tqdm(pending, desc="cosyvoice2"):
            asr = asr_rows[item.utt_id]
            output_wav = (args.output_dir / "wavs" / f"{item.utt_id}.wav").resolve()
            row = {
                "utt_id": item.utt_id,
                "speaker_id": item.speaker_id,
                "source_wav": str(item.source_wav),
                "reference_text": item.reference_text,
                "asr_raw_text": str(asr.get("asr_raw_text") or ""),
                "asr_norm_text": str(asr.get("asr_norm_text") or ""),
                "generated_wav": str(output_wav),
                "prompt_wav": str(prompt_wav),
                "prompt_text": prompt_text,
                "condition": "C0",
                "status": "success",
                "error": "",
            }
            if not row["asr_norm_text"]:
                write_silence(output_wav)
                row["status"] = "asr_failed" if asr.get("status") == "asr_failed" else "empty_asr"
                row["error"] = str(asr.get("error") or "")
            else:
                try:
                    synthesize_zero_shot(cosyvoice, row["asr_norm_text"], prompt_text, prompt_wav, output_wav)
                    if not math.isfinite(wav_duration(output_wav)) or wav_duration(output_wav) <= 0:
                        raise RuntimeError("CosyVoice2 produced non-positive duration")
                except Exception as error:  # noqa: BLE001
                    write_silence(output_wav)
                    row["status"] = "tts_failed"
                    row["error"] = truncate_error(f"{type(error).__name__}: {error}")
                    print(traceback.format_exc(), flush=True)
            append_csv(manifest, row, MANIFEST_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_baselines/asr_tts/cascade"))
    parser.add_argument("--stage", choices=["asr", "tts", "all"], default="all")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cosyvoice-repo", type=Path, default=None)
    parser.add_argument("--cosyvoice-model-dir", type=Path, default=None)
    parser.add_argument("--prompt-wav", type=Path, default=None)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt-text", default=None)
    prompt_group.add_argument("--prompt-text-file", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-batch-size", type=int, default=16)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    items = load_manifest(args.test_manifest.resolve())
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("no test items selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "run_config.json",
        {
            "condition": "C0",
            "asr_model": {"id": WAV2VEC2_MODEL_ID, "revision": WAV2VEC2_MODEL_REVISION, "decoder": "greedy CTC", "external_language_model": None},
            "tts_model": {"id": COSYVOICE_MODEL_ID, "revision": COSYVOICE_MODEL_REVISION, "source_revision": COSYVOICE_SOURCE_REVISION},
            "test_manifest": "<provided>",
            "checkpoint": "<provided>" if args.checkpoint else None,
            "cosyvoice_repo": "<provided>" if args.cosyvoice_repo else None,
            "cosyvoice_model_dir": "<provided>" if args.cosyvoice_model_dir else None,
            "prompt_wav": "<provided>" if args.prompt_wav else None,
            "prompt_text": "<provided>" if args.prompt_text or args.prompt_text_file else None,
            "prompt_text_length": len(read_prompt_text(args)) if args.stage in {"tts", "all"} else None,
            "selected_count": len(items),
            "fp16": args.fp16,
        },
    )
    if args.stage in {"asr", "all"}:
        run_asr(args, items)
    if args.stage in {"tts", "all"}:
        run_tts(args, items, read_prompt_text(args))


if __name__ == "__main__":
    main()
