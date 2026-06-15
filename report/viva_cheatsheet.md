# Viva Cheat Sheet

## One-line project intro
We built a privacy-preserving data mining pipeline that generates synthetic medical records using a class-conditional Gaussian copula model and validates utility and privacy quantitatively.

## Why this project
Real medical data cannot be shared freely due to patient privacy and legal constraints, so synthetic data enables safer collaboration.

## Core workflow
1. Generate realistic base medical tabular data.
2. Train conditional generative model (separate per disease class).
3. Create synthetic records.
4. Evaluate utility and privacy.

## Utility evidence
- Overall utility score: 0.9817
- Correlation similarity: 0.9837
- Downstream AUC (real-train): 0.789
- Downstream AUC (synthetic-train): 0.791

## Privacy evidence
- Exact match rate: 0.0
- Disclosure risk @ distance < 0.06: 0.1688
- Quasi-identifier uniqueness: 1.0

## Key claim
Synthetic data keeps major mining patterns and predictive behavior close to real data while lowering direct patient record exposure.

## Limitations
- Current implementation is tabular only.
- Differential privacy is not yet enforced during model training.
- External GAN models (CTGAN/TVAE) are listed as future work.

## Future work
1. Integrate CTGAN/TVAE with torch/SDV.
2. Add differential privacy (DP-SGD).
3. Extend to multi-hospital and multimodal settings.