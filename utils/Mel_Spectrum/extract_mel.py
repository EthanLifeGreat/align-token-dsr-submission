"""
Extract log-mel spectrograms for a prepared dataset.
Outputs to: data/<dataset>/mel/<spk_id>/<wav_id>.npy by default.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_fft=2048, win_length=2048,
                 hop_length=320, n_mels=128, norm="slaney"):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            norm=norm,
            mel_scale="slaney"
        )

    def forward(self, wav):
        # wav: [B, samples] or [samples]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = self.mel_transform(wav)  # [B, n_mels, T]
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel


def extract_mel_for_dataset(data_dir, meta_csv, mel_output_dir):
    """Extract mel spectrograms for all samples in meta.csv."""
    meta = pd.read_csv(meta_csv, dtype={'wav_id': str, 'spk_id': str})
    print(f"Total samples: {len(meta)}")

    mel_extractor = LogMelSpectrogram()
    mel_extractor.eval()

    os.makedirs(mel_output_dir, exist_ok=True)

    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        wav_id = row['wav_id']
        spk_id = row['spk_id']

        wav_path = Path(data_dir) / 'wav' / spk_id / f'{wav_id}.wav'
        mel_dir = Path(mel_output_dir) / spk_id
        mel_path = mel_dir / f'{wav_id}.npy'

        # For DEBUG
        # print(mel_path)
        # break

        # if mel_path.exists():
        #     continue

        try:
            waveform, sr = torchaudio.load(wav_path)
            # Resample to 16000 if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            with torch.no_grad():
                log_mel = mel_extractor(waveform)  # [1, 128, T]
            log_mel = log_mel.squeeze(0).numpy()  # [128, T]

            os.makedirs(mel_path.parent, exist_ok=True)
            np.save(mel_path, log_mel)
        except Exception as e:
            print(f"Error processing {wav_id}: {e}")

    print(f"Mel spectrograms saved to {mel_output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract mel spectrograms from given dataset')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name under data/. Used when --data_dir is not set.')
    parser.add_argument('--project-root', type=str, default=None,
                        help='Repository root. Defaults to the current working directory.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root data directory containing meta.csv and wav/.')
    parser.add_argument('--mel_output_dir', type=str, default=None,
                        help='Output directory for mel spectrograms.')
    args = parser.parse_args()

    project_root = Path(args.project_root or os.getcwd())
    if args.data_dir is None:
        if args.dataset is None:
            raise ValueError("Either --data_dir or --dataset is required")
        data_dir = project_root / "data" / args.dataset
    else:
        data_dir = Path(args.data_dir)
    mel_output_dir = Path(args.mel_output_dir) if args.mel_output_dir else data_dir / "mel"
    meta_csv = data_dir / 'meta.csv'
    extract_mel_for_dataset(data_dir, meta_csv, mel_output_dir)


if __name__ == '__main__':
    main()
