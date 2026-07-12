# Data Formats

## Manifest

`data/<dataset>/meta.csv` must contain:

```csv
wav_id,spk_id,split,text
utt000001,spk001,train,dummy transcript
```

`split` is one of `train`, `val`, or `test`. The dummy examples under
`examples/manifests/` show the schema only and are not derived from restricted
data.

## Waveforms

Runtime waveforms are expected at:

```text
data/<dataset>/wav/<spk_id>/<wav_id>.wav
```

Raw speech data is not included.

## External Baseline Manifests

The ASR--TTS preparation script accepts three separate CSV manifests, one for
each of train, validation, and test. Each input manifest uses this schema:

```csv
utt_id,speaker_id,source_wav,text
dummy_train_001,speaker_train_a,/path/to/legal_audio.wav,dummy transcript
```

`utt_id` must be unique across all three splits. `speaker_id` must also be
disjoint across the splits. The preparation step checks waveform existence and
writes local normalized copies with `reference_text`, `normalized_text`, and
`split` fields. Test text is retained only for later evaluation; it is never
used as ASR--TTS synthesis input.

The generated ASR--TTS manifest adds `asr_raw_text`, `asr_norm_text`,
`generated_wav`, C0 prompt fields, `status`, and `error`. The Seed-VC manifest
adds the normal target WAV, fixed conversion parameters, `status`, and `error`.
These manifests are result artifacts and are not included in the repository.

## Dense Phoneme Frames

Alignment-supervised frontend targets and outputs use one phoneme ID per frame:

```text
data/<dataset>/phoneme/<spk_id>/<wav_id>.phone
```

Each line stores one integer phoneme ID. Repetitions are retained to preserve
the dense frame interface.

## Blank-Retained CTC Frames

The CTC ablation uses framewise argmax output at the original frontend frame
rate. Blank IDs are retained and greedy transcript collapse is not used.

## Semantic Speech Tokens

CosyVoice2 semantic tokens are stored as NumPy arrays at:

```text
data/<dataset>/token/s3tokenizer_v2_25hz/<spk_id>/<wav_id>.npy
```

Token arrays are not included because they may be dataset-derived artifacts.
Use `utils/Tokenizer/s3tokenizer` to produce these files from legally obtained
audio and an externally installed tokenizer.

## Mel Spectrograms

Phoneme-to-mel and HiFi-GAN training use 16 kHz, 128-bin log-mel files:

```text
data/<dataset>/mel/<spk_id>/<wav_id>.npy
```

Use `utils/Mel_Spectrum` to extract them when offline mel files are needed.
Mel dumps are generated artifacts and are not included.

## Speaker Embeddings

Speaker embeddings are expected at:

```text
data/<dataset>/xvector/<backend>/<spk_id>/<wav_id>.npy
```

or, for speaker-level anchors:

```text
data/<dataset>/xvector/<backend>/<spk_id>.npy
```

Real embeddings and enrollment prompts are not included.

## Vocoder Checkpoints

HiFi-GAN training writes checkpoints under an untracked result directory such
as:

```text
results/vocoder_training/hifigan/bshall_legacy_matched/
```

Extracted generator weights should be kept under an untracked path such as:

```text
checkpoints/hifigan/generator-csmsc.pt
```
