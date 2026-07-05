#!/usr/bin/env python
"""Build phoneme2token length-diff and valid-frame CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def canonical_wav_id(value: object) -> str:
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text


def wav_id_candidates(value: object) -> list[str]:
    text = str(value).strip()
    values = [text]
    if text.isdigit():
        for item in (f"{int(text):06d}", str(int(text))):
            if item not in values:
                values.append(item)
    return values


def find_token_path(dataset_root: Path, token_dir_name: str, spk_id: str, wav_id: object) -> Path | None:
    for candidate in wav_id_candidates(wav_id):
        path = dataset_root / "token" / token_dir_name / spk_id / f"{candidate}.npy"
        if path.exists():
            return path
    return None


def build_filter(args: argparse.Namespace) -> None:
    dataset_root = Path(args.data_dir) / args.dataset
    frame_diff = pd.read_csv(args.frame_diff_csv)
    rows: list[dict] = []
    missing = 0
    invalid = 0
    max_abs_diff = 0
    min_diff = 0
    max_diff = 0
    examples: list[dict] = []

    for index, item in enumerate(frame_diff.itertuples(index=False), start=1):
        if args.limit is not None and index > args.limit:
            break
        if args.progress_every and index % args.progress_every == 0:
            print(f"Processed {index} rows...", flush=True)
        wav_id = getattr(item, "wav_id")
        spk_id = str(getattr(item, "spk_id")).strip()
        phone_frames_ds2 = int(getattr(item, "phone_frames_ds2"))
        token_path = find_token_path(dataset_root, args.token_dir_name, spk_id, wav_id)
        if token_path is None:
            missing += 1
            continue

        token_len = int(np.load(token_path, mmap_mode="r").reshape(-1).shape[0])
        base_len = round(phone_frames_ds2 / 2)
        diff = token_len - base_len
        abs_diff = abs(diff)
        is_valid = int(abs_diff <= args.tolerance)
        if not is_valid:
            invalid += 1
            if len(examples) < args.num_examples:
                examples.append(
                    {
                        "wav_id": wav_id,
                        "spk_id": spk_id,
                        "phone_frames_ds2": phone_frames_ds2,
                        "token_len": token_len,
                        "base_len": base_len,
                        "diff": diff,
                    }
                )
        max_abs_diff = max(max_abs_diff, abs_diff)
        min_diff = min(min_diff, diff)
        max_diff = max(max_diff, diff)
        rows.append(
            {
                "wav_id": canonical_wav_id(wav_id),
                "spk_id": spk_id,
                "phone_frames": int(getattr(item, "phone_frames")),
                "phone_frames_ds2": phone_frames_ds2,
                "phoneme_50hz_len": phone_frames_ds2,
                "token_len": token_len,
                "base_len": base_len,
                "train_target_len": base_len + args.pad_margin,
                "diff": diff,
                "abs_diff": abs_diff,
                "is_valid": is_valid,
            }
        )

    if not rows:
        raise RuntimeError("No rows were produced; check dataset/token paths.")

    output_diff = Path(args.output_diff)
    output_valid = Path(args.output_valid)
    output_diff.parent.mkdir(parents=True, exist_ok=True)
    output_valid.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with output_diff.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with output_valid.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if row["is_valid"])

    outlier_rate = invalid / len(rows)
    print("=" * 60)
    print("phoneme2token length filter")
    print("=" * 60)
    print(f"Dataset:      {args.dataset}")
    print(f"Rows:         {len(rows)}")
    print(f"Missing:      {missing}")
    print(f"Invalid:      {invalid} ({outlier_rate:.6%})")
    print(f"Tolerance:    +/-{args.tolerance}")
    print(f"Pad margin:   +{args.pad_margin}")
    print(f"Diff min/max: {min_diff}/{max_diff}")
    print(f"Max abs diff: {max_abs_diff}")
    print(f"Diff CSV:     {output_diff}")
    print(f"Valid CSV:    {output_valid}")
    if outlier_rate > args.warn_outlier_rate:
        print(
            f"WARNING: outlier rate {outlier_rate:.6%} exceeds "
            f"{args.warn_outlier_rate:.6%}"
        )
        print("First invalid examples:")
        for example in examples:
            print(example)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frame-diff-csv", required=True, type=Path)
    parser.add_argument("--token-dir-name", default="s3tokenizer_v2_25hz")
    parser.add_argument("--output-diff", required=True)
    parser.add_argument("--output-valid", required=True)
    parser.add_argument("--tolerance", default=2, type=int)
    parser.add_argument("--pad-margin", default=2, type=int)
    parser.add_argument("--warn-outlier-rate", default=0.001, type=float)
    parser.add_argument("--num-examples", default=10, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--progress-every", default=50000, type=int)
    args = parser.parse_args()
    build_filter(args)


if __name__ == "__main__":
    main()
