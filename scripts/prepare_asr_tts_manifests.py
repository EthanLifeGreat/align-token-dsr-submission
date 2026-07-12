#!/usr/bin/env python
"""Validate and normalize legal train/validation/test manifests for ASR--TTS."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.external_baselines.common import (
    COSYVOICE_MODEL_ID,
    COSYVOICE_MODEL_REVISION,
    WAV2VEC2_MODEL_ID,
    WAV2VEC2_MODEL_REVISION,
    load_manifest,
    manifest_rows,
    validate_disjoint_splits,
    write_csv,
    write_json,
)


FIELDS = ["utt_id", "speaker_id", "source_wav", "reference_text", "normalized_text", "split"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/external_baselines/asr_tts/prepared"))
    args = parser.parse_args()

    splits = {
        "train": load_manifest(args.train_manifest.resolve()),
        "validation": load_manifest(args.validation_manifest.resolve()),
        "test": load_manifest(args.test_manifest.resolve()),
    }
    validate_disjoint_splits(splits)
    output_dir = args.output_dir.resolve()
    for split, items in splits.items():
        write_csv(output_dir / f"{split}.csv", manifest_rows(items, split), FIELDS)

    summary = {
        "split_counts": {split: len(items) for split, items in splits.items()},
        "speaker_counts": {split: len({item.speaker_id for item in items}) for split, items in splits.items()},
        "speaker_disjoint": True,
        "utterance_disjoint": True,
        "normalization": "retain Hanzi, ASCII letters, and digits; remove punctuation and whitespace",
        "base_asr_model": {"id": WAV2VEC2_MODEL_ID, "revision": WAV2VEC2_MODEL_REVISION},
        "tts_model": {"id": COSYVOICE_MODEL_ID, "revision": COSYVOICE_MODEL_REVISION},
    }
    write_json(output_dir / "data_split_audit.json", summary)
    print(f"Prepared manifests in {output_dir}")


if __name__ == "__main__":
    main()
