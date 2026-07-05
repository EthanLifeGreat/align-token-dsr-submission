"""Extract Mel Spectrograms (Worker version for parallel processing).

This script is designed for parallel processing across multiple GPUs.
Each worker processes a subset of data based on worker_id.

Usage:
    CUDA_VISIBLE_DEVICES=0 python extract_mel_worker.py \
        --num-workers 8 \
        --worker-id 0 \
        --dataset LibriTTS
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


class LogMelSpectrogram(torch.nn.Module):
    """Log Mel Spectrogram extractor."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 320,
        n_mels: int = 128,
        norm: str = "slaney",
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            norm=norm,
            mel_scale="slaney",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract log mel spectrogram.

        Args:
            wav: Waveform tensor [B, samples] or [samples]

        Returns:
            Log mel spectrogram [B, n_mels, T] or [n_mels, T]
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel_transform(wav)  # [B, n_mels, T]
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel


def get_audio_paths(
    project_root: str,
    dataset: str,
    split: Optional[str] = None,
) -> List[Tuple[Path, str, str]]:
    """Get all audio file paths for a dataset.

    Args:
        project_root: Project root path
        dataset: Dataset name
        split: Data split to process (None for all)

    Returns:
        List of (audio_path, spk_id, wav_id) tuples
    """
    data_dir = Path(project_root) / "data" / dataset
    meta_path = data_dir / "meta.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.csv not found: {meta_path}")

    meta_df = pd.read_csv(meta_path, dtype={"wav_id": str})

    # Filter by split
    if split and "split" in meta_df.columns:
        meta_df = meta_df[meta_df["split"] == split]

    audio_files = []
    for _, row in meta_df.iterrows():
        wav_id = row["wav_id"]
        spk_id = row["spk_id"]

        # wav_id format: "chapter_id/basename" (e.g., "90543/7190_90543_000016_000000")
        # audio_path: wav/{spk_id}/{chapter_id}/{basename}.wav
        if "/" in wav_id:
            chapter_id, basename = wav_id.split("/", 1)
            audio_path = data_dir / "wav" / spk_id / chapter_id / f"{basename}.wav"
        else:
            audio_path = data_dir / "wav" / spk_id / f"{wav_id}.wav"

        if not audio_path.exists():
            # Try alternative extensions
            for ext in [".flac", ".mp3", ".m4a"]:
                alt_path = audio_path.with_suffix(ext)
                if alt_path.exists():
                    audio_path = alt_path
                    break
            else:
                continue

        audio_files.append((audio_path, spk_id, wav_id))

    return audio_files


def extract_mel_worker(
    mel_extractor: LogMelSpectrogram,
    audio_files: List[Tuple[Path, str, str]],
    output_dir: Path,
    overwrite: bool = False,
) -> int:
    """Extract mel spectrograms for a subset of audio files.

    Args:
        mel_extractor: LogMelSpectrogram extractor
        audio_files: List of (audio_path, spk_id, wav_id) for this worker
        output_dir: Output directory
        overwrite: Whether to overwrite existing files

    Returns:
        Number of successfully processed files
    """
    success_count = 0

    for audio_path, spk_id, wav_id in tqdm(audio_files, desc="Extracting mel"):
        out_path = output_dir / spk_id / f"{wav_id}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            success_count += 1
            continue

        try:
            waveform, sr = torchaudio.load(audio_path)
            # Resample to 16000 if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            with torch.no_grad():
                # Move mel_extractor to the correct device in main()
                waveform = waveform.to(next(mel_extractor.parameters()).device)
                log_mel = mel_extractor(waveform)  # [1, 128, T]
                log_mel = log_mel.cpu()
            log_mel = log_mel.squeeze(0).numpy()  # [128, T]

            np.save(str(out_path), log_mel)
            success_count += 1
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    return success_count


def main():
    parser = argparse.ArgumentParser(description="Extract Mel Spectrograms (Worker)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (LibriTTS, MAGICDATA, etc.)")
    parser.add_argument("--split", type=str, default=None,
                        help="Data split to process (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/{dataset}/mel)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Sample rate (default: 16000)")
    parser.add_argument("--n-fft", type=int, default=2048,
                        help="FFT size (default: 2048)")
    parser.add_argument("--win-length", type=int, default=2048,
                        help="Window length (default: 2048)")
    parser.add_argument("--hop-length", type=int, default=320,
                        help="Hop length (default: 320)")
    parser.add_argument("--n-mels", type=int, default=128,
                        help="Number of mel bands (default: 128)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (GPU is set via CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Project root path")
    # Worker-specific arguments
    parser.add_argument("--num-workers", type=int, required=True,
                        help="Total number of workers")
    parser.add_argument("--worker-id", type=int, required=True,
                        help="Worker ID (0 to num_workers-1)")

    args = parser.parse_args()

    # Validate worker_id
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError(f"worker_id must be in [0, {args.num_workers-1}], got {args.worker_id}")

    # Get project root
    if args.project_root:
        project_root = Path(args.project_root)
    else:
        # Script is at: project_root/utils/Mel_Spectrum/extract_mel_worker.py
        # Need to go up 3 levels: Mel_Spectrum -> utils -> project_root
        project_root = Path(__file__).parent.parent.parent

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "data" / args.dataset / "mel"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Worker {args.worker_id}/{args.num_workers}]")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output directory: {output_dir}")
    print(f"  Mel params: sr={args.sample_rate}, n_fft={args.n_fft}, n_mels={args.n_mels}")
    print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # Create mel extractor
    mel_extractor = LogMelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    )
    mel_extractor = mel_extractor.to(args.device)
    mel_extractor.eval()

    # Get audio files
    audio_files = get_audio_paths(str(project_root), args.dataset, args.split)
    print(f"  Total audio files: {len(audio_files)}")

    # Filter by worker_id: process files where i % num_workers == worker_id
    audio_files = [f for i, f in enumerate(audio_files) if i % args.num_workers == args.worker_id]
    print(f"  This worker processes: {len(audio_files)} files")

    # Extract mel spectrograms
    success_count = extract_mel_worker(
        mel_extractor, audio_files, output_dir, overwrite=args.overwrite
    )

    print(f"  Successfully extracted: {success_count}/{len(audio_files)} files")


if __name__ == "__main__":
    main()