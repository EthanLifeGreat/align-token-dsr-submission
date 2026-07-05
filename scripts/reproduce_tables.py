"""Aggregate prepared result CSV files into compact table CSVs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cer-csv", type=Path, required=True, help="CSV with method,N,S,D,I columns.")
    parser.add_argument("--quality-csv", type=Path, default=None, help="Optional CSV with method,utmos,dnsmos columns.")
    parser.add_argument("--speaker-csv", type=Path, default=None, help="Optional CSV with method,similarity columns.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.cer_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (args.output_dir / "table_cer_edit_rates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "CER", "sub_rate", "del_rate", "ins_rate"])
        writer.writeheader()
        for row in rows:
            n = max(float(row["N"]), 1.0)
            s, d, i = float(row["S"]), float(row["D"]), float(row["I"])
            writer.writerow({
                "method": row["method"],
                "CER": f"{(s + d + i) / n:.6f}",
                "sub_rate": f"{s / n:.6f}",
                "del_rate": f"{d / n:.6f}",
                "ins_rate": f"{i / n:.6f}",
            })

    for optional_csv, output_name, columns in [
        (args.quality_csv, "table_predicted_quality.csv", ["utmos", "dnsmos"]),
        (args.speaker_csv, "table_speaker_similarity.csv", ["similarity"]),
    ]:
        if optional_csv is None:
            continue
        sums: dict[str, dict[str, float]] = defaultdict(lambda: {col: 0.0 for col in columns} | {"count": 0.0})
        with optional_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bucket = sums[row["method"]]
                bucket["count"] += 1.0
                for col in columns:
                    bucket[col] += float(row[col])
        with (args.output_dir / output_name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", *columns])
            writer.writeheader()
            for method, values in sorted(sums.items()):
                count = max(values.pop("count"), 1.0)
                writer.writerow({"method": method, **{col: f"{values[col] / count:.6f}" for col in columns}})


if __name__ == "__main__":
    main()
