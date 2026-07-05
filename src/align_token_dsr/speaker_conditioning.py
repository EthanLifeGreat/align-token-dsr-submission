"""Speaker-conditioning helpers for AlignToken-DSR.

The paper uses three conditioning choices:

* C0: a fixed normal-speaker anchor selected by configuration.
* C1: the nearest normal-speaker anchor by cosine similarity.
* C2: a source-identity enrollment anchor.

This module operates only on caller-provided embeddings and paths. It does not
ship real speaker embeddings or prompt audio.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AnchorSelection:
    condition: str
    speaker_id: str
    embedding_path: Path | None = None
    prompt_wav_path: Path | None = None
    score: float | None = None


def _load_embedding(path: Path) -> np.ndarray:
    value = np.load(path).astype(np.float32).reshape(-1)
    norm = np.linalg.norm(value)
    if norm <= 0:
        raise ValueError(f"zero-norm embedding: {path}")
    return value / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def select_c0(
    fixed_speaker_id: str,
    normal_anchor_embeddings: Mapping[str, Path],
    prompt_wavs: Mapping[str, Path] | None = None,
) -> AnchorSelection:
    if fixed_speaker_id not in normal_anchor_embeddings:
        raise KeyError(f"missing C0 anchor embedding for speaker {fixed_speaker_id}")
    prompt_wavs = prompt_wavs or {}
    return AnchorSelection(
        condition="C0",
        speaker_id=fixed_speaker_id,
        embedding_path=normal_anchor_embeddings[fixed_speaker_id],
        prompt_wav_path=prompt_wavs.get(fixed_speaker_id),
    )


def select_c1(
    source_embedding_path: Path,
    normal_anchor_embeddings: Mapping[str, Path],
    prompt_wavs: Mapping[str, Path] | None = None,
) -> AnchorSelection:
    source = _load_embedding(source_embedding_path)
    if not normal_anchor_embeddings:
        raise ValueError("normal_anchor_embeddings must not be empty")
    best_speaker = ""
    best_path: Path | None = None
    best_score = -2.0
    for speaker_id, path in normal_anchor_embeddings.items():
        score = _cosine(source, _load_embedding(path))
        if score > best_score:
            best_speaker = speaker_id
            best_path = path
            best_score = score
    prompt_wavs = prompt_wavs or {}
    return AnchorSelection(
        condition="C1",
        speaker_id=best_speaker,
        embedding_path=best_path,
        prompt_wav_path=prompt_wavs.get(best_speaker),
        score=best_score,
    )


def select_c2(
    source_speaker_id: str,
    enrollment_embedding_path: Path,
    enrollment_prompt_wav_path: Path | None = None,
) -> AnchorSelection:
    return AnchorSelection(
        condition="C2",
        speaker_id=source_speaker_id,
        embedding_path=enrollment_embedding_path,
        prompt_wav_path=enrollment_prompt_wav_path,
    )

