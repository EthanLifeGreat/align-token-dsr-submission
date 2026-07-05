#!/usr/bin/env python3

import argparse
from pathlib import Path, PurePosixPath

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Campplus speaker embeddings from kaldi .pt files to .npy files."
    )
    parser.add_argument(
        "--kaldi-dir",
        type=Path,
        required=True,
        help="Kaldi directory containing spk2embedding.pt, utt2embedding.pt and utt2spk.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory, e.g. data/<dataset>/xvector/Campplus.",
    )
    return parser.parse_args()


def load_mapping(path: Path):
    mapping = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def normalize_embedding(embedding):
    return np.asarray(embedding, dtype=np.float32)


def speaker_path(output_dir: Path, speaker_id: str) -> Path:
    speaker_parts = PurePosixPath(speaker_id).parts
    if not speaker_parts:
        raise ValueError("Empty speaker id is not allowed.")
    if len(speaker_parts) == 1:
        return output_dir / f"{speaker_parts[0]}.npy"
    return output_dir.joinpath(*speaker_parts[:-1], f"{speaker_parts[-1]}.npy")


def utterance_path(output_dir: Path, speaker_id: str, utterance_id: str) -> Path:
    utt_name = PurePosixPath(utterance_id).name
    if not utt_name:
        raise ValueError(f"Invalid utterance id: {utterance_id}")
    speaker_parts = PurePosixPath(speaker_id).parts
    return output_dir.joinpath(*speaker_parts, f"{utt_name}.npy")


def save_embeddings(kaldi_dir: Path, output_dir: Path):
    spk_path = kaldi_dir / "spk2embedding.pt"
    utt_path = kaldi_dir / "utt2embedding.pt"
    utt2spk_path = kaldi_dir / "utt2spk"

    for required_path in (spk_path, utt_path, utt2spk_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required file not found: {required_path}")

    spk2embedding = torch.load(spk_path, map_location="cpu", weights_only=False)
    utt2embedding = torch.load(utt_path, map_location="cpu", weights_only=False)
    utt2spk = load_mapping(utt2spk_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    spk_count = 0
    for speaker_id, embedding in spk2embedding.items():
        target_path = speaker_path(output_dir, speaker_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, normalize_embedding(embedding))
        spk_count += 1

    utt_count = 0
    missing_spk = []
    for utterance_id, embedding in utt2embedding.items():
        speaker_id = utt2spk.get(utterance_id)
        if speaker_id is None:
            inferred_speaker = str(PurePosixPath(utterance_id).parent)
            if inferred_speaker and inferred_speaker != ".":
                speaker_id = inferred_speaker
            else:
                missing_spk.append(utterance_id)
                continue

        target_path = utterance_path(output_dir, speaker_id, utterance_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, normalize_embedding(embedding))
        utt_count += 1

    if missing_spk:
        preview = ", ".join(missing_spk[:5])
        raise ValueError(
            f"Failed to resolve speaker ids for {len(missing_spk)} utterances: {preview}"
        )

    print(f"Saved {spk_count} speaker embeddings to {output_dir}")
    print(f"Saved {utt_count} utterance embeddings to {output_dir}")


def main():
    args = parse_args()
    save_embeddings(args.kaldi_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
