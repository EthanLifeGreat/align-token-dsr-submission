# Reproduction Notes

1. Obtain legal access to the datasets and external pretrained renderers.
2. Prepare manifests and features using `docs/data_formats.md`.
3. Extract CosyVoice semantic tokens with `utils/Tokenizer/s3tokenizer`.
4. Extract mel features with `utils/Mel_Spectrum` when offline mel files are
   needed.
5. Train HiFi-GAN with `utils/Vocoder/hifigan/train/run.sh`, then extract
   generator weights with `utils/Vocoder/hifigan/run_extract_weights.sh`.
6. Train or place other checkpoints under untracked `checkpoints/`.
7. Run the frontend, token route, Token2Wav renderer, acoustic-reference route,
   and evaluation wrappers.
8. Reproduce tables from prepared result CSVs with `scripts/reproduce_tables.py`.

The paper tables can be regenerated without committing private artifacts by
using aggregate prepared result files with the schemas shown in
`examples/results/`.

## External Baselines

The ASR--TTS and Seed-VC workflows are documented separately in
`docs/external_baselines.md`. In brief:

1. Validate legal train, validation, and test manifests with
   `scripts/prepare_asr_tts_manifests.py`.
2. Fine-tune the pinned Wav2Vec2-CTC checkpoint using only the prepared train
   and validation manifests.
3. Decode test audio with greedy CTC and render only the decoded text through
   CosyVoice2 using a user-provided C0 prompt pair.
4. Run Seed-VC through a user-provided persistent JSONL worker; the runner
   sends only source/target audio paths and fixed conversion parameters.
5. Export evaluation-ASR hypotheses, then use `scripts/compute_cer.py`,
   `utils/Naturalness/scripts/evaluate_naturalness.py`,
   `scripts/compute_speaker_similarity.py`, and
   `scripts/aggregate_baseline_metrics.py`.

Generated manifests, models, audio, worker logs, and metrics are written under
ignored result directories and must not be committed.
