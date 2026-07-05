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

Use placeholder paths such as `checkpoints/wav2phoneme/AISHELL2-FA` in configs.
Before releasing any checkpoint, verify that the source dataset license and all
third-party model licenses permit redistribution.

CDSD-adapted checkpoints and CDSD-derived artifacts must be treated as
restricted unless a separate license review explicitly permits release.
