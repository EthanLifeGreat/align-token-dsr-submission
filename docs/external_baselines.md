# External Baseline Reproduction

This repository provides reproduction code for two paper baselines. It does
not contain source audio, prompts, trained ASR checkpoints, generated audio,
model files, or evaluation outputs.

## Fixed Public Dependencies

| Component | Public identifier | Fixed revision |
| --- | --- | --- |
| Base CTC ASR | [`qinyue/wav2vec2-large-xlsr-53-chinese-zn-cn-aishell1`](https://huggingface.co/qinyue/wav2vec2-large-xlsr-53-chinese-zn-cn-aishell1) | `6486b012e64ea1014cd8c91e98779823656eb28a` |
| CosyVoice source | [`FunAudioLLM/CosyVoice`](https://github.com/FunAudioLLM/CosyVoice) | `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc` |
| CosyVoice model | [`FunAudioLLM/CosyVoice2-0.5B`](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | `eec1ae6c79877dbd9379285cf8789c9e0879293d` |
| Seed-VC source | [`Plachtaa/seed-vc`](https://github.com/Plachtaa/seed-vc) v2.0 | `8555549e882236e6541748b1042d95693caa82ba` |
| Seed-VC model | [`Plachta/Seed-VC`](https://huggingface.co/Plachta/Seed-VC) | `257283f9f41585055e8f858fba4fd044e5caed6e` |

Install external source repositories and model files outside this checkout. The
ASR--TTS runner verifies the CosyVoice Git revision before synthesis; check out
the exact source revision above rather than using an archive copy. The Seed-VC
source is GPL-3.0; no Seed-VC source code or adapter is copied here.

## ASR--TTS Baseline

The paper condition trains Wav2Vec2-CTC on the legal train split, chooses a
checkpoint using validation corpus CER only, applies greedy CTC decoding with
no language model on test audio, and renders predicted normalized text with
CosyVoice2-0.5B using a canonical normal C0 prompt pair. Test transcripts are
not ASR--TTS inference inputs.

Input manifests use the schema in `docs/data_formats.md`. The CSVs under
`examples/manifests/` show only field names and placeholder values.

```bash
python scripts/prepare_asr_tts_manifests.py \
  --train-manifest /path/to/train.csv \
  --validation-manifest /path/to/validation.csv \
  --test-manifest /path/to/test.csv \
  --output-dir results/external_baselines/asr_tts/prepared

python scripts/train_wav2vec2_ctc.py \
  --prepared-dir results/external_baselines/asr_tts/prepared \
  --output-dir results/external_baselines/asr_tts/finetune \
  --device cuda

python scripts/run_wav2vec2_ctc_cosyvoice2.py \
  --stage all \
  --test-manifest results/external_baselines/asr_tts/prepared/test.csv \
  --checkpoint "$(cat results/external_baselines/asr_tts/finetune/best_checkpoint.txt)" \
  --cosyvoice-repo /path/to/CosyVoice \
  --cosyvoice-model-dir /path/to/CosyVoice2-0.5B \
  --prompt-wav /path/to/c0_prompt.wav \
  --prompt-text-file /path/to/c0_prompt_text.txt \
  --output-dir results/external_baselines/asr_tts/cascade \
  --device cuda
```

The cascade manifest retains every selected utterance. Empty ASR output writes
a 0.5-second silence with `empty_asr`; ASR exceptions use `asr_failed`; TTS
exceptions use `tts_failed`. Each case retains its error string.

## Seed-VC Normal-Speaker Reference

This baseline converts each source waveform to the same user-provided normal
target waveform. It is text-free: reference text is copied to the local result
manifest only for later CER evaluation and is never sent to Seed-VC.

The runner requires a persistent external worker. Its standard output must be
JSON Lines only; diagnostics must go to standard error. For each request, it
must write one response before reading the next request.

### Request

```json
{
  "protocol_version": "1",
  "utt_id": "example_id",
  "source_wav": "/path/to/source.wav",
  "target_wav": "/path/to/c0_target.wav",
  "output_wav": "/path/to/output.wav",
  "seed_vc_model": "Seed-VC-v2-hubert-bsqvae-small",
  "seed_vc_source_revision": "8555549e882236e6541748b1042d95693caa82ba",
  "seed_vc_model_id": "Plachta/Seed-VC",
  "seed_vc_model_revision": "257283f9f41585055e8f858fba4fd044e5caed6e",
  "diffusion_steps": 25,
  "length_adjust": 1.0,
  "intelligibility_cfg_rate": 0.7,
  "similarity_cfg_rate": 0.7,
  "fp16": true,
  "convert_style": false,
  "anonymization_only": false
}
```

### Response

```json
{
  "utt_id": "example_id",
  "status": "success",
  "error": "",
  "seed_vc_model": "Seed-VC-v2-hubert-bsqvae-small",
  "seed_vc_source_revision": "8555549e882236e6541748b1042d95693caa82ba",
  "seed_vc_model_revision": "257283f9f41585055e8f858fba4fd044e5caed6e"
}
```

The runner rejects a response with an unexpected ID, status, model, or revision.
It writes a 0.5-second silence with `vc_failed` when the worker fails, times
out, returns invalid JSON, or does not create a valid waveform. The worker
command is intentionally not written to the run configuration.

```bash
python scripts/run_seed_vc_normal_ref.py \
  --test-manifest /path/to/test.csv \
  --target-wav /path/to/c0_target.wav \
  --seed-vc-worker '/path/to/seed_vc_v2_jsonl_worker' \
  --output-dir results/external_baselines/seed_vc_normal_ref
```

## Evaluation and Aggregation

Use the same external ASR evaluation configuration for every system, export
Kaldi-style reference and hypothesis text, then calculate CER and edit counts:

```bash
python scripts/compute_cer.py \
  --ref /path/to/reference.kaldi.txt \
  --hyp /path/to/evaluation_asr_hypothesis.kaldi.txt \
  --output results/external_baselines/asr_tts/cer_summary.csv \
  --per-utt-output results/external_baselines/asr_tts/cer_per_utt.csv

python utils/Naturalness/scripts/evaluate_naturalness.py \
  --wav-dir results/external_baselines/asr_tts/cascade/wavs \
  --output-csv results/external_baselines/asr_tts/naturalness.csv \
  --device cuda

python scripts/aggregate_baseline_metrics.py \
  --manifest results/external_baselines/asr_tts/cascade/manifest.csv \
  --method asr_tts_c0 \
  --cer-per-utt-csv results/external_baselines/asr_tts/cer_per_utt.csv \
  --quality-csv results/external_baselines/asr_tts/naturalness.csv \
  --speaker-csv /path/to/prepared_campp_scores.csv \
  --output-dir results/external_baselines/asr_tts/metrics
```

`scripts/compute_speaker_similarity.py` consumes prepared embedding pairs and
can produce the `source_sim` or `cond_sim` fields used by the aggregation step.
All commands write data-derived artifacts below ignored `results/` paths.
