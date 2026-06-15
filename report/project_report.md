# Project Report

## Title
Generative AI-Based Synthetic Medical Data Generation for Privacy Preservation

## Abstract
This project develops a privacy-preserving synthetic medical data generation pipeline using a class-conditional Gaussian copula model. The pipeline generates realistic synthetic patient records from source medical tabular data and evaluates both utility and privacy. Utility is measured through distribution similarity, correlation preservation, and downstream classification efficacy. Privacy is measured through exact match leakage, nearest-neighbor disclosure risk, and quasi-identifier uniqueness. The final results demonstrate that synthetic data can retain mining value while reducing direct patient-level disclosure risk.

## Problem Statement
Medical datasets are highly sensitive and often unavailable for open data mining. A practical approach is required to share data patterns for model development without exposing actual patient records.

## Objectives
1. Build a synthetic medical data generator based on generative modeling.
2. Evaluate utility of synthetic data for mining tasks.
3. Evaluate privacy leakage risk after generation.
4. Deliver a reproducible end-to-end pipeline.

## Methodology
1. Generate realistic baseline medical tabular dataset.
2. Split into train/test for unbiased downstream evaluation.
3. Train class-conditional Gaussian copula synthesizer on real train data.
4. Generate synthetic dataset of equal size.
5. Evaluate utility and privacy metrics.
6. Compare downstream model performance:
   - Train on real, test on real.
   - Train on synthetic, test on real.

## Experimental Setup
- Real rows (total/train/test): 2200/1760/440
- Synthetic rows: 1760
- Runtime timestamp: 2026-03-15 20:05:02

## Results
### Utility
- Overall utility score: **0.9817**
- Continuous similarity: **0.9768**
- Categorical similarity: **0.9911**
- Correlation similarity: **0.9837**

### Downstream efficacy (Logistic Regression)
- Train Real -> Test Real: accuracy=0.7636, precision=0.6944, recall=0.5137, f1=0.5906, auc=0.789
- Train Synth -> Test Real: accuracy=0.7682, precision=0.7245, recall=0.4863, f1=0.582, auc=0.791
- Delta (Synth - Real): accuracy=0.0046, precision=0.0301, recall=-0.0274, f1=-0.0086, auc=0.002

### Privacy
- Overall privacy score: **0.5275**
- Exact match rate: **0.0**
- Disclosure risk at distance < 0.06: **0.1688**
- Nearest-neighbor distance (p01/p05/median): **0.0355 / 0.048 / 0.0834**
- Quasi-identifier uniqueness: **1.0**

## Conclusion
The project demonstrates that synthetic medical data generation can preserve core data mining utility while improving privacy protection compared to direct sharing of original records. This provides a practical privacy-preserving pathway for collaborative analytics and educational model development.

## Tools and Technology
- Python (standard library)
- CSV-based data pipeline
- Custom implementations for copula synthesis, logistic regression, and evaluation metrics

## Future Enhancements
1. Add GAN-based models (CTGAN/TVAE) when external ML libraries are available.
2. Add formal differential privacy training.
3. Extend to image and multi-modal medical datasets.
