# Project modifications

This directory vendors code derived from
[`bshall/hifigan`](https://github.com/bshall/hifigan).

Project-specific changes include:

- `meta.csv` datasets with `wav/<spk_id>/<wav_id>.wav` path resolution;
- the AlignToken-DSR legacy 16 kHz, 128-bin, power-2 centered log-mel contract;
- AISHELL-2 pretraining followed by CSMSC fine-tuning;
- true DDP validation sharding with global loss reduction;
- validation subsets, maximum steps, and early stopping;
- optimizer reset for fine-tuning;
- retention of only `latest.pt` and `model-best.pt`.

The upstream license is retained in `LICENSE`.
