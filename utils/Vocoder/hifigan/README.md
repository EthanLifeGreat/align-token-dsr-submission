# HiFi-GAN Vocoder

This directory contains the HiFi-GAN inference wrapper used by the
phoneme-to-mel acoustic reference route, plus the training scripts used to
prepare the 16 kHz, 128-bin log-mel vocoder.

No vocoder checkpoints or extracted generator weights are committed.

## Training

The training entrypoint wraps the vendored MIT-licensed `bshall/hifigan` source
under `utils/Vocoder/hifigan/src/bshall/`.

```bash
HIFIGAN_GPUS=0,1 bash utils/Vocoder/hifigan/train/run.sh pretrain
HIFIGAN_GPUS=0,1 bash utils/Vocoder/hifigan/train/run.sh finetune
```

or run both stages:

```bash
HIFIGAN_GPUS=0,1 bash utils/Vocoder/hifigan/train/run.sh all
```

Defaults:

- pretrain data: `data/AISHELL-2-95_5/meta.csv` and `data/AISHELL-2-95_5/wav`;
- finetune data: `data/CSMSC/meta.csv` and `data/CSMSC/wav`;
- output: `results/vocoder_training/hifigan/bshall_legacy_matched/`;
- conda environment: `hifigan`.

Useful overrides:

```bash
CONDA_BIN=/path/to/conda \
HIFIGAN_CONDA_ENV=hifigan \
HIFIGAN_GPUS=0,1 \
HIFIGAN_PRETRAIN_DIR=results/vocoder_training/hifigan/pretrain \
HIFIGAN_FINETUNE_DIR=results/vocoder_training/hifigan/finetune \
bash utils/Vocoder/hifigan/train/run.sh all
```

## Extract Generator Weights

After training, extract only generator weights for inference:

```bash
HIFIGAN_CHECKPOINT_DIR=results/vocoder_training/hifigan/bshall_legacy_matched/finetune_csmsc \
HIFIGAN_OUTPUT_WEIGHTS=checkpoints/hifigan/generator-csmsc.pt \
bash utils/Vocoder/hifigan/run_extract_weights.sh
```

The extracted generator file is still a model artifact and should stay outside
git unless redistribution is explicitly permitted.

## Inference

```python
from utils.Vocoder.hifigan import HiFiGANAPI

vocoder = HiFiGANAPI(checkpoint_path="checkpoints/hifigan/generator-csmsc.pt")
wav, sample_rate = vocoder.inference(mel)
```

The wrapper accepts mel arrays shaped either `[time, mel_bins]` or
`[mel_bins, time]` and validates the mel dimension from the checkpoint.
