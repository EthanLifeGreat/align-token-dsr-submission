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
