# Mel Spectrum Extraction

These scripts extract the 16 kHz, 128-bin log-mel spectrograms used by the
phoneme-to-mel acoustic reference route and by HiFi-GAN vocoder training.

## Sequential Extraction

```bash
python utils/Mel_Spectrum/extract_mel.py \
  --dataset CSMSC \
  --project-root "$PWD"
```

This writes:

```text
data/CSMSC/mel/<spk_id>/<wav_id>.npy
```

## Parallel Extraction

```bash
bash utils/Mel_Spectrum/extract_mel_parallel.sh "0,1,2,3" 1 CSMSC
```

Use `MEL_CONDA_ENV` and `CONDA_BIN` to select the runtime environment:

```bash
MEL_CONDA_ENV=dsr CONDA_BIN=conda \
bash utils/Mel_Spectrum/extract_mel_parallel.sh "0,1" 1 AISHELL-2-95_5
```

Mel files are generated dataset-derived artifacts and are intentionally not
committed.
