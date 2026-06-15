# Generative AI-Based Synthetic Medical Data Generation for Privacy Preservation

A complete, self-contained Data Mining project implemented with Python standard library only.

## What this project does

1. Creates a realistic **medical tabular dataset** (`real` data).
2. Trains a **class-conditional Gaussian copula synthesizer** to generate synthetic patient records.
3. Evaluates:
   - Utility (distribution similarity, correlation preservation, downstream ML efficacy),
   - Privacy (duplicate leakage, nearest-neighbor disclosure risk, uniqueness).
4. Produces ready-to-submit outputs in `results/` and a report in `report/`.

## Why this fits Data Mining

- Uses mining workflows: preprocessing, pattern modeling, utility evaluation, classification performance.
- Focuses on privacy-preserving data sharing while preserving data usefulness.

## Project structure

```
CEP/
  data/
    raw/
    synthetic/
  report/
    project_report.md
  results/
    metrics.json
    summary.txt
  src/
    dataset.py
    synthesizer.py
    metrics.py
    model.py
    pipeline.py
  run_project.py
```

## Run (one command)

```powershell
python run_project.py
```

## Main outputs

- `data/raw/real_medical_data.csv`
- `data/synthetic/synthetic_medical_data.csv`
- `results/metrics.json`
- `results/summary.txt`

## Notes

- No external libraries required.
- The generated records are synthetic and non-identifying.
