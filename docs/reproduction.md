# Reproduction Notes

1. Obtain legal access to the datasets and external pretrained renderers.
2. Prepare manifests and features using `docs/data_formats.md`.
3. Train or place checkpoints under untracked `checkpoints/`.
4. Run the frontend, token route, Token2Wav renderer, and evaluation wrappers.
5. Reproduce tables from prepared result CSVs with `scripts/reproduce_tables.py`.

The paper tables can be regenerated without committing private artifacts by
using aggregate prepared result files with the schemas shown in
`examples/results/`.

