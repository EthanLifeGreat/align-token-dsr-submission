"""Compute character error rate and edit-operation rates from Kaldi text files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_kaldi_text(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt_id, _, text = line.partition(" ")
        rows[utt_id] = text.replace(" ", "")
    return rows


def _edit_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    n = len(ref)
    dp = [[(0, 0, 0, 0) for _ in range(len(hyp) + 1)] for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        s, d, ins, cost = dp[i - 1][0]
        dp[i][0] = (s, d + 1, ins, cost + 1)
    for j in range(1, len(hyp) + 1):
        s, d, ins, cost = dp[0][j - 1]
        dp[0][j] = (s, d, ins + 1, cost + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            choices: list[tuple[int, int, int, int]] = []
            s, d, ins, cost = dp[i - 1][j - 1]
            choices.append((s, d, ins, cost) if ref[i - 1] == hyp[j - 1] else (s + 1, d, ins, cost + 1))
            s, d, ins, cost = dp[i - 1][j]
            choices.append((s, d + 1, ins, cost + 1))
            s, d, ins, cost = dp[i][j - 1]
            choices.append((s, d, ins + 1, cost + 1))
            dp[i][j] = min(choices, key=lambda item: (item[3], item[0] + item[1] + item[2]))
    return dp[-1][-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=Path, required=True, help="Kaldi-style reference text: <utt_id> <text>")
    parser.add_argument("--hyp", type=Path, required=True, help="Kaldi-style hypothesis text: <utt_id> <text>")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-utt-output", type=Path, default=None, help="Optional per-utterance edit-count CSV.")
    args = parser.parse_args()

    ref_rows = _read_kaldi_text(args.ref)
    hyp_rows = _read_kaldi_text(args.hyp)
    shared = sorted(set(ref_rows) & set(hyp_rows))
    if not shared:
        raise ValueError("no shared utterance ids between reference and hypothesis")

    total_n = total_s = total_d = total_i = 0
    per_utt_rows = []
    for utt_id in shared:
        ref = ref_rows[utt_id]
        hyp = hyp_rows[utt_id]
        s, d, ins, _ = _edit_counts(ref, hyp)
        total_n += len(ref)
        total_s += s
        total_d += d
        total_i += ins
        per_utt_rows.append({
            "utt_id": utt_id,
            "N": len(ref),
            "S": s,
            "D": d,
            "I": ins,
            "CER": f"{(s + d + ins) / max(len(ref), 1):.6f}",
        })
    cer = (total_s + total_d + total_i) / max(total_n, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "utterances,N,S,D,I,CER,sub_rate,del_rate,ins_rate\n"
        f"{len(shared)},{total_n},{total_s},{total_d},{total_i},"
        f"{cer:.6f},{total_s / max(total_n, 1):.6f},"
        f"{total_d / max(total_n, 1):.6f},{total_i / max(total_n, 1):.6f}\n",
        encoding="utf-8",
    )
    if args.per_utt_output is not None:
        args.per_utt_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_utt_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["utt_id", "N", "S", "D", "I", "CER"])
            writer.writeheader()
            writer.writerows(per_utt_rows)


if __name__ == "__main__":
    main()
