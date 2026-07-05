"""Compute cosine speaker similarity from prepared embedding pairs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load(path: Path) -> np.ndarray:
    value = np.load(path).astype(np.float32).reshape(-1)
    norm = np.linalg.norm(value)
    if norm <= 0:
        raise ValueError(f"zero-norm embedding: {path}")
    return value / norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", type=Path, required=True, help="CSV with utt_id,method,ref_embedding,hyp_embedding")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    with args.pairs_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref = _load(Path(row["ref_embedding"]))
            hyp = _load(Path(row["hyp_embedding"]))
            rows.append({
                "utt_id": row["utt_id"],
                "method": row["method"],
                "similarity": f"{float(np.dot(ref, hyp)):.6f}",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "method", "similarity"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
