#!/usr/bin/env python
"""Run the Seed-VC v2 normal-speaker reference through an external JSONL worker."""
from __future__ import annotations

import argparse
import csv
import json
import select
import shlex
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.external_baselines.common import (
    SEED_VC_MODEL_ID,
    SEED_VC_MODEL_NAME,
    SEED_VC_MODEL_REVISION,
    SEED_VC_PROTOCOL_VERSION,
    SEED_VC_SOURCE_REVISION,
    append_csv,
    build_seed_vc_request,
    load_manifest,
    read_csv,
    truncate_error,
    validate_seed_vc_response,
    wav_duration,
    write_json,
    write_silence,
)


FIELDS = [
    "utt_id", "speaker_id", "source_wav", "reference_text", "target_wav", "generated_wav", "status", "error",
    "condition",
    "seed_vc_model", "seed_vc_source_revision", "seed_vc_model_revision", "diffusion_steps", "intelligibility_cfg_rate",
    "similarity_cfg_rate", "length_adjust", "fp16", "convert_style", "anonymization_only",
]


def existing_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["utt_id"]: row for row in csv.DictReader(handle) if row.get("utt_id")}


def read_worker_response(worker: subprocess.Popen[str], timeout_sec: float) -> dict[str, object]:
    if worker.stdout is None:
        raise RuntimeError("worker stdout pipe is unavailable")
    ready, _, _ = select.select([worker.stdout], [], [], timeout_sec)
    if not ready:
        raise TimeoutError(f"worker did not respond within {timeout_sec:g} seconds")
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError(f"worker exited before responding (exit_code={worker.poll()})")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"worker stdout is not one JSON object per line: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("worker response must be a JSON object")
    return value


def request_conversion(worker: subprocess.Popen[str], request: dict[str, object], timeout_sec: float) -> str:
    if worker.stdin is None:
        return "worker stdin pipe is unavailable"
    try:
        worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        worker.stdin.flush()
        response = read_worker_response(worker, timeout_sec)
    except Exception as error:  # noqa: BLE001
        return truncate_error(f"{type(error).__name__}: {error}")
    return validate_seed_vc_response(response, str(request["utt_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--target-wav", type=Path, required=True)
    parser.add_argument("--seed-vc-worker", required=True, help="Quoted external worker command; never stored in run_config.json.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_baselines/seed_vc_normal_ref"))
    parser.add_argument("--worker-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    target_wav = args.target_wav.resolve()
    if not target_wav.is_file():
        raise FileNotFoundError(target_wav)
    items = load_manifest(args.test_manifest.resolve())
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise ValueError("no test items selected")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.csv"
    if manifest.exists() and not args.resume:
        raise FileExistsError(f"Refusing to append to existing {manifest}; use --resume or a new output directory")
    existing = existing_rows(manifest) if args.resume else {}
    pending = [item for item in items if item.utt_id not in existing]
    write_json(
        output_dir / "run_config.json",
        {
            "condition": "C0 normal target",
            "worker_protocol_version": SEED_VC_PROTOCOL_VERSION,
            "seed_vc_worker": "configured externally (redacted)",
            "seed_vc_source_revision": SEED_VC_SOURCE_REVISION,
            "seed_vc_model": SEED_VC_MODEL_NAME,
            "seed_vc_model_id": SEED_VC_MODEL_ID,
            "seed_vc_model_revision": SEED_VC_MODEL_REVISION,
            "diffusion_steps": 25,
            "intelligibility_cfg_rate": 0.7,
            "similarity_cfg_rate": 0.7,
            "length_adjust": 1.0,
            "fp16": True,
            "convert_style": False,
            "anonymization_only": False,
            "test_manifest": "<provided>",
            "target_wav": "<provided>",
            "selected_count": len(items),
        },
    )
    stderr_path = output_dir / "logs" / "seed_vc_worker.stderr.log"
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stderr_path.open("a", encoding="utf-8") as stderr:
        worker = subprocess.Popen(
            shlex.split(args.seed_vc_worker),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        try:
            for item in tqdm(pending, desc="seed-vc"):
                output_wav = (output_dir / "wavs" / f"{item.utt_id}.wav").resolve()
                request = build_seed_vc_request(item, target_wav, output_wav)
                error = request_conversion(worker, request, args.worker_timeout_sec)
                if not error and (not output_wav.is_file() or wav_duration(output_wav) <= 0):
                    error = "worker reported success but did not write a valid waveform"
                row = {
                    "utt_id": item.utt_id,
                    "speaker_id": item.speaker_id,
                    "source_wav": str(item.source_wav),
                    "reference_text": item.reference_text,
                    "target_wav": str(target_wav),
                    "generated_wav": str(output_wav),
                    "status": "success" if not error else "vc_failed",
                    "error": truncate_error(error),
                    "condition": "C0",
                    "seed_vc_model": SEED_VC_MODEL_NAME,
                    "seed_vc_source_revision": SEED_VC_SOURCE_REVISION,
                    "seed_vc_model_revision": SEED_VC_MODEL_REVISION,
                    "diffusion_steps": 25,
                    "intelligibility_cfg_rate": 0.7,
                    "similarity_cfg_rate": 0.7,
                    "length_adjust": 1.0,
                    "fp16": True,
                    "convert_style": False,
                    "anonymization_only": False,
                }
                if error:
                    write_silence(output_wav)
                append_csv(manifest, row, FIELDS)
        finally:
            if worker.stdin is not None:
                worker.stdin.close()
            try:
                worker.wait(timeout=30)
            except subprocess.TimeoutExpired:
                worker.terminate()
                worker.wait(timeout=30)


if __name__ == "__main__":
    main()
