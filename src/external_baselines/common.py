"""Shared, path-agnostic helpers for external baseline scripts."""
from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import soundfile as sf


WAV2VEC2_MODEL_ID = "qinyue/wav2vec2-large-xlsr-53-chinese-zn-cn-aishell1"
WAV2VEC2_MODEL_REVISION = "6486b012e64ea1014cd8c91e98779823656eb28a"
COSYVOICE_SOURCE_REVISION = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
COSYVOICE_MODEL_ID = "FunAudioLLM/CosyVoice2-0.5B"
COSYVOICE_MODEL_REVISION = "eec1ae6c79877dbd9379285cf8789c9e0879293d"
SEED_VC_SOURCE_REVISION = "8555549e882236e6541748b1042d95693caa82ba"
SEED_VC_MODEL_ID = "Plachta/Seed-VC"
SEED_VC_MODEL_REVISION = "257283f9f41585055e8f858fba4fd044e5caed6e"
SEED_VC_MODEL_NAME = "Seed-VC-v2-hubert-bsqvae-small"
SEED_VC_PROTOCOL_VERSION = "1"

ASR_TEXT_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")
CER_TEXT_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class ManifestItem:
    utt_id: str
    speaker_id: str
    source_wav: Path
    reference_text: str


def normalize_asr_text(text: str) -> str:
    """Retain Hanzi, ASCII letters, and digits while removing punctuation."""
    return "".join(ASR_TEXT_RE.findall(str(text or "").strip()))


def normalize_cer_text(text: str) -> str:
    """Use the Chinese-character normalization used by the paper CER protocol."""
    return "".join(CER_TEXT_RE.findall(str(text or "").strip()))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(materialized[0]) if materialized else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def append_csv(path: Path, row: Mapping[str, object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def wav_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return math.nan
    return float(info.frames / info.samplerate) if info.samplerate > 0 else math.nan


def write_silence(path: Path, duration_sec: float = 0.5, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(max(1, int(duration_sec * sample_rate)), dtype=np.float32), sample_rate)


def _required_value(row: Mapping[str, str], field: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"manifest row is missing {field!r}: {dict(row)}")
    return value


def load_manifest(path: Path, *, require_audio: bool = True) -> list[ManifestItem]:
    """Load the public baseline manifest schema without assuming a dataset layout."""
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    items: list[ManifestItem] = []
    seen: set[str] = set()
    for row in rows:
        utt_id = _required_value(row, "utt_id")
        if utt_id in seen:
            raise ValueError(f"duplicate utt_id {utt_id!r} in {path}")
        seen.add(utt_id)
        speaker_id = _required_value(row, "speaker_id")
        source_wav = Path(_required_value(row, "source_wav")).expanduser()
        if not source_wav.is_absolute():
            source_wav = (path.parent / source_wav).resolve()
        if require_audio and not source_wav.is_file():
            raise FileNotFoundError(f"missing source_wav for {utt_id}: {source_wav}")
        reference_text = str(row.get("reference_text") or row.get("text") or "")
        items.append(ManifestItem(utt_id, speaker_id, source_wav, reference_text))
    return items


def validate_disjoint_splits(splits: Mapping[str, list[ManifestItem]]) -> None:
    """Reject overlap between train, validation, and test utterances or speakers."""
    names = list(splits)
    for index, left_name in enumerate(names):
        left = splits[left_name]
        left_utts = {item.utt_id for item in left}
        left_speakers = {item.speaker_id for item in left}
        for right_name in names[index + 1 :]:
            right = splits[right_name]
            duplicate_utts = left_utts & {item.utt_id for item in right}
            duplicate_speakers = left_speakers & {item.speaker_id for item in right}
            if duplicate_utts:
                raise ValueError(
                    f"utterance overlap between {left_name} and {right_name}: "
                    f"{sorted(duplicate_utts)[:5]}"
                )
            if duplicate_speakers:
                raise ValueError(
                    f"speaker overlap between {left_name} and {right_name}: "
                    f"{sorted(duplicate_speakers)[:5]}"
                )


def manifest_rows(items: Iterable[ManifestItem], split: str) -> list[dict[str, str]]:
    return [
        {
            "utt_id": item.utt_id,
            "speaker_id": item.speaker_id,
            "source_wav": str(item.source_wav),
            "reference_text": item.reference_text,
            "normalized_text": normalize_asr_text(item.reference_text),
            "split": split,
        }
        for item in items
    ]


def edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """Return reference length and substitution/deletion/insertion counts."""
    reference = normalize_cer_text(reference)
    hypothesis = normalize_cer_text(hypothesis)
    dp = [[(0, 0, 0, 0) for _ in range(len(hypothesis) + 1)] for _ in range(len(reference) + 1)]
    for i in range(1, len(reference) + 1):
        sub, dele, ins, cost = dp[i - 1][0]
        dp[i][0] = (sub, dele + 1, ins, cost + 1)
    for j in range(1, len(hypothesis) + 1):
        sub, dele, ins, cost = dp[0][j - 1]
        dp[0][j] = (sub, dele, ins + 1, cost + 1)
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            sub, dele, ins, cost = dp[i - 1][j - 1]
            substitution = (sub, dele, ins, cost) if reference[i - 1] == hypothesis[j - 1] else (sub + 1, dele, ins, cost + 1)
            sub, dele, ins, cost = dp[i - 1][j]
            deletion = (sub, dele + 1, ins, cost + 1)
            sub, dele, ins, cost = dp[i][j - 1]
            insertion = (sub, dele, ins + 1, cost + 1)
            dp[i][j] = min((substitution, deletion, insertion), key=lambda value: (value[3], value[0] + value[1] + value[2]))
    sub, dele, ins, _ = dp[-1][-1]
    return len(reference), sub, dele, ins


def aggregate_edit_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, float | int]:
    materialized = list(rows)
    ref_chars = sum(int(row.get("N", row.get("ref_chars", 0)) or 0) for row in materialized)
    substitutions = sum(int(row.get("S", row.get("substitutions", 0)) or 0) for row in materialized)
    deletions = sum(int(row.get("D", row.get("deletions", 0)) or 0) for row in materialized)
    insertions = sum(int(row.get("I", row.get("insertions", 0)) or 0) for row in materialized)
    denominator = max(ref_chars, 1)
    return {
        "N": len(materialized),
        "ref_chars": ref_chars,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "cer_pct": 100.0 * (substitutions + deletions + insertions) / denominator,
        "sub_rate": substitutions / denominator,
        "del_rate": deletions / denominator,
        "ins_rate": insertions / denominator,
    }


def truncate_error(value: object, limit: int = 2000) -> str:
    text = str(value or "").strip()
    return text[:limit]


def verify_git_revision(repository: Path, expected_revision: str, dependency: str) -> None:
    """Require an externally installed source checkout to match its pinned revision."""
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{dependency} must be a git checkout pinned to {expected_revision}")
    actual_revision = completed.stdout.strip()
    if actual_revision != expected_revision:
        raise RuntimeError(f"{dependency} revision mismatch: expected {expected_revision}, got {actual_revision}")


def build_seed_vc_request(item: ManifestItem, target_wav: Path, output_wav: Path) -> dict[str, object]:
    """Create the documented process-level Seed-VC worker request."""
    return {
        "protocol_version": SEED_VC_PROTOCOL_VERSION,
        "utt_id": item.utt_id,
        "source_wav": str(item.source_wav),
        "target_wav": str(target_wav),
        "output_wav": str(output_wav),
        "seed_vc_model": SEED_VC_MODEL_NAME,
        "seed_vc_source_revision": SEED_VC_SOURCE_REVISION,
        "seed_vc_model_id": SEED_VC_MODEL_ID,
        "seed_vc_model_revision": SEED_VC_MODEL_REVISION,
        "diffusion_steps": 25,
        "length_adjust": 1.0,
        "intelligibility_cfg_rate": 0.7,
        "similarity_cfg_rate": 0.7,
        "fp16": True,
        "convert_style": False,
        "anonymization_only": False,
    }


def validate_seed_vc_response(response: Mapping[str, object], utt_id: str) -> str:
    """Return an empty string for a valid success response, otherwise an error."""
    if str(response.get("utt_id") or "") != utt_id:
        return "worker response utt_id does not match request"
    if str(response.get("status") or "") != "success":
        return truncate_error(response.get("error") or "worker reported failure")
    expected = {
        "seed_vc_model": SEED_VC_MODEL_NAME,
        "seed_vc_source_revision": SEED_VC_SOURCE_REVISION,
        "seed_vc_model_revision": SEED_VC_MODEL_REVISION,
    }
    for field, value in expected.items():
        if str(response.get(field) or "") != value:
            return f"worker response has unexpected {field}"
    return ""
