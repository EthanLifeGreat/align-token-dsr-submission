# HiFi-GAN training

Training uses the vendored
[`bshall/hifigan`](https://github.com/bshall/hifigan) source under
`utils/Vocoder/hifigan/src/bshall/`.

The retained `run.sh` dispatches stages to the vendored bshall-based trainer.

The reused trainer:

- reads each dataset's `meta.csv` directly;
- resolves waveforms as `wav/<spk_id>/<wav_id>.wav`;
- computes the AlignToken-DSR 16 kHz, 128-bin, power-2 centered log-mel online;
- shards validation across DDP ranks and globally reduces the loss;
- excludes the `test` split from training;
- keeps only `latest.pt` and `model-best.pt`;
- pretrains from random initialization and resets optimizer state for CSMSC.

```bash
HIFIGAN_GPUS=0,1 bash utils/Vocoder/hifigan/train/run.sh all
```

Stages are `pretrain`, `finetune`, and `all`. Outputs default to
`results/vocoder_training/hifigan/bshall_legacy_matched/`.
