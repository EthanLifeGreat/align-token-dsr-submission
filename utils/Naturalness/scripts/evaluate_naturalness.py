"""Batch objective naturalness/quality evaluation with UTMOS and DNSMOS."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NATURALNESS_ROOT = PROJECT_ROOT / "utils" / "Naturalness"
SPEECHMOS_ROOT = NATURALNESS_ROOT / "SpeechMOS"
DNSMOS_ROOT = NATURALNESS_ROOT / "DNS-Challenge" / "DNSMOS"
SAMPLE_RATE = 16000
DNSMOS_INPUT_SECONDS = 9.01


def _load_speechmos_hubconf():
    hubconf_path = SPEECHMOS_ROOT / "hubconf.py"
    spec = importlib.util.spec_from_file_location("speechmos_hubconf", hubconf_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SpeechMOS hubconf from {hubconf_path}")
    sys.path.insert(0, str(SPEECHMOS_ROOT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UTMOSScorer:
    def __init__(self, device: str) -> None:
        hubconf = _load_speechmos_hubconf()
        requested_device = torch.device(device)
        model = hubconf.utmos22_strong(pretrained=True).eval()
        try:
            self.device = requested_device
            self.model = model.to(self.device)
        except RuntimeError as exc:
            if requested_device.type != "cuda":
                raise
            print(f"CUDA unavailable for UTMOS ({exc}); falling back to CPU.", file=sys.stderr)
            self.device = torch.device("cpu")
            self.model = model.to(self.device)

    @torch.inference_mode()
    def score(self, wav_path: Path) -> float:
        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.mean(dim=0, keepdim=True).to(self.device)
        score = self.model(wav, int(sr))
        return float(score.squeeze().detach().cpu().item())

    @torch.inference_mode()
    def score_batch(self, items: list[dict[str, str]]) -> dict[str, float]:
        waves = []
        sample_rate = None
        max_len = 0
        for item in items:
            wav, sr = torchaudio.load(item["wav_path"])
            wav = wav.mean(dim=0)
            if sample_rate is None:
                sample_rate = int(sr)
            elif sample_rate != int(sr):
                raise ValueError("UTMOS batch contains mixed sample rates")
            waves.append(wav)
            max_len = max(max_len, int(wav.numel()))
        padded = []
        for wav in waves:
            if wav.numel() < max_len:
                repeats = 1 + (max_len - wav.numel()) // max(1, wav.numel())
                pad_src = wav.repeat(repeats + 1)
                wav = torch.cat([wav, pad_src[: max_len - wav.numel()]])
            padded.append(wav)
        batch = torch.stack(padded, dim=0).to(self.device)
        scores = self.model(batch, int(sample_rate)).detach().cpu().numpy()
        return {
            item["wav_id"]: float(score)
            for item, score in zip(items, scores, strict=True)
        }


class DNSMOSScorer:
    def __init__(self, personalized: bool = False) -> None:
        primary_name = "pDNSMOS/sig_bak_ovr.onnx" if personalized else "DNSMOS/sig_bak_ovr.onnx"
        primary_model_path = DNSMOS_ROOT / primary_name
        p808_model_path = DNSMOS_ROOT / "DNSMOS" / "model_v8.onnx"
        self.onnx_sess = ort.InferenceSession(str(primary_model_path))
        self.p808_onnx_sess = ort.InferenceSession(str(p808_model_path))
        self.personalized = personalized

    @staticmethod
    def _audio_melspec(
        audio: np.ndarray,
        n_mels: int = 120,
        frame_size: int = 320,
        hop_length: int = 160,
        sr: int = SAMPLE_RATE,
    ) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=frame_size + 1,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        return ((librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40).T

    def _polyfit(self, sig: float, bak: float, ovr: float) -> tuple[float, float, float]:
        if self.personalized:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovr))

    def score(self, wav_path: Path) -> dict[str, float]:
        audio, input_fs = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if input_fs != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=int(input_fs), target_sr=SAMPLE_RATE)
        audio = np.asarray(audio, dtype=np.float32)

        actual_audio_len = len(audio)
        len_samples = int(DNSMOS_INPUT_SECONDS * SAMPLE_RATE)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLE_RATE) - DNSMOS_INPUT_SECONDS) + 1
        hop_len_samples = SAMPLE_RATE
        sig_raw_values: list[float] = []
        bak_raw_values: list[float] = []
        ovr_raw_values: list[float] = []
        sig_values: list[float] = []
        bak_values: list[float] = []
        ovr_values: list[float] = []
        p808_values: list[float] = []

        for idx in range(num_hops):
            start = idx * hop_len_samples
            end = int((idx + DNSMOS_INPUT_SECONDS) * SAMPLE_RATE)
            audio_seg = audio[start:end]
            if len(audio_seg) < len_samples:
                continue
            primary_input = audio_seg.astype(np.float32)[np.newaxis, :]
            p808_input = self._audio_melspec(audio_seg[:-160]).astype(np.float32)[np.newaxis, :, :]
            p808_mos = float(self.p808_onnx_sess.run(None, {"input_1": p808_input})[0][0][0])
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(
                None, {"input_1": primary_input}
            )[0][0]
            mos_sig, mos_bak, mos_ovr = self._polyfit(
                float(mos_sig_raw),
                float(mos_bak_raw),
                float(mos_ovr_raw),
            )
            sig_raw_values.append(float(mos_sig_raw))
            bak_raw_values.append(float(mos_bak_raw))
            ovr_raw_values.append(float(mos_ovr_raw))
            sig_values.append(mos_sig)
            bak_values.append(mos_bak)
            ovr_values.append(mos_ovr)
            p808_values.append(p808_mos)

        return {
            "dnsmos_len_sec": float(actual_audio_len / SAMPLE_RATE),
            "dnsmos_num_hops": float(num_hops),
            "dnsmos_sig_raw": float(np.mean(sig_raw_values)),
            "dnsmos_bak_raw": float(np.mean(bak_raw_values)),
            "dnsmos_ovrl_raw": float(np.mean(ovr_raw_values)),
            "dnsmos_sig": float(np.mean(sig_values)),
            "dnsmos_bak": float(np.mean(bak_values)),
            "dnsmos_ovrl": float(np.mean(ovr_values)),
            "dnsmos_p808_mos": float(np.mean(p808_values)),
        }


def load_items(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.manifest:
        items = []
        with Path(args.manifest).open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                obj = json.loads(line)
                wav_path = obj.get("output_path") or obj.get("wav_path")
                if not wav_path:
                    continue
                items.append(
                    {
                        "wav_id": str(obj.get("wav_id") or Path(wav_path).stem),
                        "spk_id": str(obj.get("spk_id") or Path(wav_path).parent.name),
                        "wav_path": str(wav_path),
                    }
                )
    else:
        wav_dir = Path(args.wav_dir)
        items = [
            {
                "wav_id": path.stem,
                "spk_id": path.parent.name,
                "wav_path": str(path),
            }
            for path in sorted(wav_dir.rglob("*.wav"))
        ]
    if args.limit is not None:
        items = items[: args.limit]
    return items


def read_done_wav_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    try:
        df = pd.read_csv(output_csv)
    except pd.errors.EmptyDataError:
        return set()
    if "wav_id" not in df.columns:
        return set()
    return set(df["wav_id"].astype(str).tolist())


def append_rows(output_csv: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def batched_by_sample_rate(
    items: list[dict[str, str]],
    batch_size: int,
) -> Iterable[list[dict[str, str]]]:
    batch: list[dict[str, str]] = []
    batch_sr: int | None = None
    for item in items:
        try:
            info = torchaudio.info(item["wav_path"])
            sr = int(info.sample_rate)
        except Exception:
            sr = None
        if batch and (len(batch) >= batch_size or sr != batch_sr):
            yield batch
            batch = []
            batch_sr = None
        batch.append(item)
        batch_sr = sr
    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="JSONL manifest with output_path/wav_path, wav_id, spk_id.")
    source.add_argument("--wav-dir", help="Directory searched recursively for .wav files.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--errors-csv", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-utmos", action="store_true")
    parser.add_argument("--skip-dnsmos", action="store_true")
    parser.add_argument("--personalized-dnsmos", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dnsmos-workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.skip_utmos and args.skip_dnsmos:
        raise ValueError("At least one scorer must be enabled.")

    output_csv = Path(args.output_csv)
    errors_csv = Path(args.errors_csv) if args.errors_csv else output_csv.with_name("errors.csv")
    done_wav_ids = read_done_wav_ids(output_csv) if args.resume else set()
    items = [item for item in load_items(args) if item["wav_id"] not in done_wav_ids]

    utmos = None if args.skip_utmos else UTMOSScorer(device=args.device)
    dnsmos = None if args.skip_dnsmos else DNSMOSScorer(personalized=args.personalized_dnsmos)

    progress = tqdm(total=len(items), desc="naturalness")
    for chunk in batched_by_sample_rate(items, max(1, args.batch_size)):
        rows_by_id: dict[str, dict[str, object]] = {
            item["wav_id"]: {
                "wav_id": item["wav_id"],
                "spk_id": item["spk_id"],
                "wav_path": item["wav_path"],
            }
            for item in chunk
        }
        failed_ids: set[str] = set()

        if utmos is not None:
            try:
                for wav_id, score in utmos.score_batch(chunk).items():
                    rows_by_id[wav_id]["utmos"] = score
            except Exception as exc:  # noqa: BLE001
                for item in chunk:
                    failed_ids.add(item["wav_id"])
                    append_rows(
                        errors_csv,
                        [
                            {
                                "wav_id": item["wav_id"],
                                "spk_id": item["spk_id"],
                                "wav_path": item["wav_path"],
                                "error": f"UTMOS batch failed: {exc!r}",
                            }
                        ],
                    )

        if dnsmos is not None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.dnsmos_workers) as executor:
                future_to_item = {
                    executor.submit(dnsmos.score, Path(item["wav_path"])): item
                    for item in chunk
                    if item["wav_id"] not in failed_ids
                }
                for future in concurrent.futures.as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        rows_by_id[item["wav_id"]].update(future.result())
                    except Exception as exc:  # noqa: BLE001
                        failed_ids.add(item["wav_id"])
                        append_rows(
                            errors_csv,
                            [
                                {
                                    "wav_id": item["wav_id"],
                                    "spk_id": item["spk_id"],
                                    "wav_path": item["wav_path"],
                                    "error": f"DNSMOS failed: {exc!r}",
                                }
                            ],
                        )

        rows = [
            rows_by_id[item["wav_id"]]
            for item in chunk
            if item["wav_id"] not in failed_ids
        ]
        append_rows(output_csv, rows)
        progress.update(len(chunk))
    progress.close()


if __name__ == "__main__":
    main()
