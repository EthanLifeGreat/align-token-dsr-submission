# AlignToken-DSR Submission Repository

This is an anonymized minimal repository for the AlignToken-DSR paper. It
contains only the code, configuration templates, wrappers, and documentation
needed to reproduce the paper pipeline from legally obtained data manifests and
prepared result files.

No original git history, private data, generated audio, speaker embeddings,
prompt waveforms, pretrained model files, or adapted checkpoints are included.

## Included Components

- `src/wav2phoneme`: alignment-supervised waveform-to-phoneme frontend and
  blank-retained CTC-frame frontend ablation.
- `src/phoneme2token`: decoder-only Transformer that maps dense phoneme frames
  plus speaker embedding to CosyVoice2 semantic speech tokens.
- `src/align_token_dsr/token2wav_client.py`: client for an external frozen
  CosyVoice2 Token2Wav renderer.
- `src/align_token_dsr/speaker_conditioning.py`: C0/C1/C2 speaker-conditioning
  anchor selection logic.
- `src/phoneme2mel`: phoneme-to-mel acoustic reference route.
- `utils/CosyVoice_Token_Services`: thin wrapper for running Token2Wav without
  vendoring CosyVoice2.
- `utils/WER`, `scripts/compute_cer.py`, `scripts/reproduce_tables.py`, and
  `utils/Naturalness/scripts`: evaluation wrappers for CER, edit rates,
  predicted quality, and table reproduction.

## Data Layout

After obtaining legal access to the required corpora, prepare data under
`data/<dataset>/` using the schema in `docs/data_formats.md`. The repository
expects prepared features only at runtime and does not include restricted
artifacts.

## Training Examples

Alignment-supervised frontend:

```bash
PYTHONPATH=$PWD bash src/wav2phoneme/train.sh AISHELL2-FA
PYTHONPATH=$PWD bash src/wav2phoneme/train.sh CDSD-FA
```

Blank-retained CTC ablation:

```bash
PYTHONPATH=$PWD bash src/wav2phoneme/train.sh AISHELL2-CTC
PYTHONPATH=$PWD bash src/wav2phoneme/train.sh CDSD-CTC
```

Phoneme-to-semantic-token decoder:

```bash
PYTHONPATH=$PWD bash src/phoneme2token/train.sh AISHELL-2-95_5
PYTHONPATH=$PWD bash src/phoneme2token/train.sh CSMSC-finetune
```

Phoneme-to-mel acoustic reference:

```bash
PYTHONPATH=$PWD bash src/phoneme2mel/train.sh rope192_smallpostnet/AISHELL-2-pretrain
PYTHONPATH=$PWD bash src/phoneme2mel/train.sh rope192_smallpostnet/CSMSC-finetune
```

## Rendering and Evaluation

Start the external CosyVoice2 Token2Wav service as described in
`utils/CosyVoice_Token_Services/README.md`, then render tokens:

```bash
PYTHONPATH=$PWD python -m src.align_token_dsr.token2wav_client \
  --token-npy /path/to/predicted_tokens.npy \
  --prompt-wav /path/to/legal_prompt.wav \
  --output-wav /path/to/output.wav
```

Compute CER and edit rates from prepared ASR hypotheses:

```bash
python scripts/compute_cer.py \
  --ref examples/results/ref.kaldi.txt \
  --hyp examples/results/hyp.kaldi.txt \
  --output results/eval/cer_edit_rates.csv
```

Reproduce aggregate tables from prepared result files:

```bash
python scripts/reproduce_tables.py \
  --cer-csv examples/results/prepared_cer_counts.csv \
  --quality-csv examples/results/prepared_quality_scores.csv \
  --speaker-csv examples/results/prepared_speaker_similarity.csv \
  --output-dir results/tables
```

## Artifact Policy

See `docs/checkpoint_policy.md`. All private or license-restricted checkpoints,
CDSD-derived labels, speaker embeddings, enrollment prompts, and generated
reconstruction samples must remain outside this repository.

