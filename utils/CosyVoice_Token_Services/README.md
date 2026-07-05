# CosyVoice2 Token2Wav Wrapper

This directory contains a thin service wrapper for rendering predicted
CosyVoice2 semantic speech tokens into waveforms. The submission repository does
not vendor CosyVoice2 source code, pretrained weights, checkpoints, prompt
audio, or generated samples.

## External Setup

Install CosyVoice2 from its upstream project and obtain the pretrained
Token2Wav model according to its license. Then start this service with explicit
paths:

```bash
export PYTHONPATH=/path/to/CosyVoice:/path/to/CosyVoice/third_party/Matcha-TTS:$PWD
python utils/CosyVoice_Token_Services/server.py \
  --cosyvoice-model-dir /path/to/CosyVoice2-0.5B \
  --host 127.0.0.1 \
  --port 8000 \
  --device cuda:0
```

The renderer is treated as fixed and pretrained. AlignToken-DSR training does
not update CosyVoice2 parameters.

## Token2Wav Request

The service accepts a JSON request with semantic token IDs and an optional
prompt waveform, depending on the conditioning route:

```json
{
  "token_ids": [12, 42, 108],
  "prompt_wav_base64": "<optional base64 wav>",
  "prompt_text": "",
  "speed": 1.0
}
```

For double-blind submission, use anonymous placeholder paths in scripts and do
not commit prompt waveforms, speaker embeddings, pretrained model files, or
generated audio.
