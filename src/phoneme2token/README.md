# phoneme2token

Decoder-only Transformer for:

```text
dense phoneme sequence + speaker embedding -> CosyVoice2 semantic tokens
```

The model uses a projected speaker embedding and dense frame-level phoneme
prefix, then autoregressively predicts a fixed-length semantic-token sequence.
The output length is set from the known phoneme-frame to semantic-token frame
rate ratio, with a small padding margin configured by `length_pad_margin`.

## Training

```bash
PYTHONPATH=$PWD bash src/phoneme2token/train.sh AISHELL-2-95_5
PYTHONPATH=$PWD bash src/phoneme2token/train.sh CSMSC-finetune
```

## Inference

```bash
PYTHONPATH=$PWD python src/phoneme2token/inference.py \
  --checkpoint checkpoints/phoneme2token/CSMSC-finetune \
  --phoneme-path /path/to/dense.phone \
  --xvector-path /path/to/speaker.npy \
  --output-token-path /path/to/predicted_tokens.npy
```

## Evaluation Convention

CER and edit-rate reports should use Kaldi-style text files:

```text
utt_id reference text
utt_id hypothesis text
```

Then run:

```bash
python scripts/compute_cer.py \
  --ref examples/results/ref.kaldi.txt \
  --hyp examples/results/hyp.kaldi.txt \
  --output results/eval/cer_edit_rates.csv
```
