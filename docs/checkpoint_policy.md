# Checkpoint and Artifact Policy

This repository intentionally excludes:

- raw speech data;
- forced-alignment files derived from restricted corpora;
- dense frame labels derived from restricted corpora;
- CosyVoice2 pretrained model files;
- S3Tokenizer pretrained model files and extracted token arrays;
- extracted mel-spectrogram dumps;
- frontend, decoder, acoustic-reference, or vocoder checkpoints trained or
  adapted on restricted data;
- speaker embeddings and enrollment prompt audio;
- generated reconstruction samples;
- experiment logs and tracking artifacts.

This also includes Wav2Vec2-CTC checkpoints produced by the ASR--TTS training
workflow, local CosyVoice2 and Seed-VC model files, C0 prompt WAV/text pairs,
and generated baseline manifests or metrics. The code records public model IDs
and revisions in `docs/external_baselines.md`; users obtain and keep all model
files and run outputs locally.

Use placeholder paths such as `checkpoints/wav2phoneme/AISHELL2-FA` in configs.
Before releasing any checkpoint, verify that the source dataset license and all
third-party model licenses permit redistribution.

Data-derived checkpoints and artifacts produced by any baseline workflow must
remain outside this repository unless their redistribution has been reviewed
separately.
