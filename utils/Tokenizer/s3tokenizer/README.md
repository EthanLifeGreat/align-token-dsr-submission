# S3Tokenizer Pipeline

This utility extracts CosyVoice semantic speech tokens used by the
`phoneme2token` route. For the paper experiments, the relevant tokenizer is:

```text
speech_tokenizer_v2_25hz -> s3tokenizer_v2_25hz
```

The tokenizer package and pretrained tokenizer weights are external
dependencies. They are not vendored in this repository.

## Input

Each dataset must follow the repository data schema:

```text
data/<dataset>/meta.csv
data/<dataset>/wav/<spk_id>/<wav_id>.wav
```

The pipeline first builds or reuses:

```text
data/<dataset>/kaldi/wav.scp
```

## Output

Final tokens are materialized as one-dimensional `int32` NumPy arrays:

```text
data/<dataset>/token/s3tokenizer_v2_25hz/<spk_id>/<wav_id>.npy
```

Intermediate shards and logs are written under:

```text
exp/s3tokenizer_v2_25hz/<dataset>/
```

These token files are dataset-derived artifacts and are intentionally not
committed.

## Usage

Run the full pipeline:

```bash
bash utils/Tokenizer/s3tokenizer/pipeline.sh \
  --model speech_tokenizer_v2_25hz \
  --datasets AISHELL-2-95_5 CSMSC \
  --gpus "0,1,2,3"
```

Run only materialization and validation when shard files already exist:

```bash
bash utils/Tokenizer/s3tokenizer/pipeline.sh \
  --model speech_tokenizer_v2_25hz \
  --datasets CSMSC \
  --steps materialize validate
```

`pipeline.py` resolves `torchrun` and `s3tokenizer` from the current `PATH`; if
they are not available, it falls back to `conda run -n <env>`. Configure the
environment with `--conda-env` if needed.
