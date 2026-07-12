#!/usr/bin/env python
"""Merge generated-baseline status with prepared CER, quality, and similarity scores."""
from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.external_baselines.common import aggregate_edit_rows, write_csv, write_json


def read_optional(path: Path | None) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"utt_id": str, "wav_id": str}) if path is not None else pd.DataFrame()


def average(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def metrics_for_group(frame: pd.DataFrame) -> dict[str, object]:
    output: dict[str, object] = {"N": int(len(frame))}
    if {"N_cer", "S", "D", "I"}.issubset(frame.columns):
        rows = [
            {"N": row["N_cer"], "S": row["S"], "D": row["D"], "I": row["I"]}
            for _, row in frame.dropna(subset=["N_cer", "S", "D", "I"]).iterrows()
        ]
        if rows:
            output.update(aggregate_edit_rows(rows))
    for column in ["utmos", "dnsmos_ovrl", "source_sim", "cond_sim"]:
        value = average(frame, column)
        if value is not None:
            output[column] = value
    output["failed_empty_count"] = int((frame["status"].fillna("") != "success").sum())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cer-per-utt-csv", type=Path, default=None)
    parser.add_argument("--quality-csv", type=Path, default=None)
    parser.add_argument("--speaker-csv", type=Path, default=None, help="Prepared CAM++ CSV with utt_id and source_sim/cond_sim.")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"utt_id": str, "speaker_id": str}, keep_default_na=False)
    required = {"utt_id", "speaker_id", "status"}
    if missing := required - set(manifest):
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if manifest["utt_id"].duplicated().any():
        raise ValueError("manifest contains duplicate utt_id values")
    merged = manifest.copy()

    cer = read_optional(args.cer_per_utt_csv)
    if not cer.empty:
        if "utt_id" not in cer:
            raise ValueError("CER CSV requires utt_id")
        if cer["utt_id"].duplicated().any():
            raise ValueError("CER CSV contains duplicate utt_id values")
        cer = cer.rename(columns={"N": "N_cer"})
        merged = merged.merge(cer, on="utt_id", how="left", validate="one_to_one")

    quality = read_optional(args.quality_csv)
    if not quality.empty:
        quality = quality.rename(columns={"wav_id": "utt_id"})
        if "utt_id" not in quality or quality["utt_id"].duplicated().any():
            raise ValueError("quality CSV requires unique wav_id or utt_id")
        quality_columns = [column for column in ["utt_id", "utmos", "dnsmos_ovrl"] if column in quality]
        merged = merged.merge(quality[quality_columns], on="utt_id", how="left", validate="one_to_one")

    similarity = read_optional(args.speaker_csv)
    if not similarity.empty:
        similarity = similarity.rename(columns={"wav_id": "utt_id"})
        if "utt_id" not in similarity or similarity["utt_id"].duplicated().any():
            raise ValueError("speaker CSV requires unique wav_id or utt_id")
        similarity_columns = [column for column in ["utt_id", "source_sim", "cond_sim"] if column in similarity]
        merged = merged.merge(similarity[similarity_columns], on="utt_id", how="left", validate="one_to_one")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "metrics_per_utt.csv", index=False)
    per_speaker = []
    for speaker_id, group in merged.groupby("speaker_id", sort=True, dropna=False):
        per_speaker.append({"speaker_id": speaker_id, **metrics_for_group(group)})
    write_csv(output_dir / "metrics_per_speaker.csv", per_speaker)
    summary = {
        "method": args.method,
        "total_count": int(len(merged)),
        "status_counts": dict(Counter(merged["status"].fillna(""))),
        **metrics_for_group(merged),
    }
    write_json(output_dir / "metrics_summary.json", summary)
    write_csv(output_dir / "metrics_summary.csv", [summary])


if __name__ == "__main__":
    main()
