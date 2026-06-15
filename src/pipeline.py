"""End-to-end pipeline for synthetic medical data project."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.dataset import (
    CATEGORICAL_COLUMNS,
    CONTINUOUS_COLUMNS,
    generate_medical_dataset,
    train_test_split,
    write_csv,
)
from src.metrics import privacy_report, utility_report
from src.model import evaluate_logistic_regression
from src.synthesizer import ConditionalGaussianCopulaSynthesizer

ROOT = Path(__file__).resolve().parents[1]
DATA_REAL_PATH = ROOT / "data" / "raw" / "real_medical_data.csv"
DATA_SYNTH_PATH = ROOT / "data" / "synthetic" / "synthetic_medical_data.csv"
METRICS_PATH = ROOT / "results" / "metrics.json"
SUMMARY_PATH = ROOT / "results" / "summary.txt"
REPORT_PATH = ROOT / "report" / "project_report.md"


def _round_dict(d: dict[str, float], places: int = 4) -> dict[str, float]:
    return {k: round(v, places) for k, v in d.items()}


def _build_summary(metrics: dict[str, Any]) -> str:
    util = metrics["utility"]
    priv = metrics["privacy"]
    real_m = metrics["downstream"]["train_real_test_real"]
    synth_m = metrics["downstream"]["train_synth_test_real"]
    delta = metrics["downstream"]["delta_synth_minus_real"]

    lines = [
        "Generative AI-Based Synthetic Medical Data Generation for Privacy Preservation",
        "=" * 78,
        "",
        f"Run timestamp: {metrics['run_timestamp']}",
        f"Real rows (total/train/test): {metrics['data']['real_total']}/{metrics['data']['real_train']}/{metrics['data']['real_test']}",
        f"Synthetic rows: {metrics['data']['synthetic_total']}",
        "",
        "UTILITY METRICS",
        "-" * 30,
        f"Overall utility score: {util['overall_utility_score']}",
        f"Continuous similarity: {util['continuous_similarity']}",
        f"Categorical similarity: {util['categorical_similarity']}",
        f"Correlation similarity: {util['correlation_similarity']}",
        "",
        "DOWNSTREAM MODEL EFFICACY (Logistic Regression)",
        "-" * 30,
        f"Train Real -> Test Real: accuracy={real_m['accuracy']}, f1={real_m['f1']}, auc={real_m['auc']}",
        f"Train Synth -> Test Real: accuracy={synth_m['accuracy']}, f1={synth_m['f1']}, auc={synth_m['auc']}",
        f"Delta (Synth - Real): accuracy={delta['accuracy']}, f1={delta['f1']}, auc={delta['auc']}",
        "",
        "PRIVACY METRICS",
        "-" * 30,
        f"Overall privacy score: {priv['overall_privacy_score']}",
        f"Exact match rate: {priv['exact_match_rate']}",
        f"Disclosure risk @ distance<0.06: {priv['disclosure_risk_at_0_06']}",
        f"NN distance p01/p05/median: {priv['nearest_neighbor_distance_p01']}/{priv['nearest_neighbor_distance_p05']}/{priv['nearest_neighbor_distance_median']}",
        f"Quasi-identifier uniqueness: {priv['quasi_identifier_uniqueness']}",
        "",
        "INTERPRETATION",
        "-" * 30,
        "Synthetic data preserves major statistical and predictive patterns while reducing direct linkage risk.",
        "This supports privacy-preserving data sharing for educational and research-focused data mining workflows.",
    ]

    return "\n".join(lines)


def _build_report_markdown(metrics: dict[str, Any]) -> str:
    util = metrics["utility"]
    priv = metrics["privacy"]
    real_m = metrics["downstream"]["train_real_test_real"]
    synth_m = metrics["downstream"]["train_synth_test_real"]
    delta = metrics["downstream"]["delta_synth_minus_real"]

    return f"""# Project Report

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
- Real rows (total/train/test): {metrics['data']['real_total']}/{metrics['data']['real_train']}/{metrics['data']['real_test']}
- Synthetic rows: {metrics['data']['synthetic_total']}
- Runtime timestamp: {metrics['run_timestamp']}

## Results
### Utility
- Overall utility score: **{util['overall_utility_score']}**
- Continuous similarity: **{util['continuous_similarity']}**
- Categorical similarity: **{util['categorical_similarity']}**
- Correlation similarity: **{util['correlation_similarity']}**

### Downstream efficacy (Logistic Regression)
- Train Real -> Test Real: accuracy={real_m['accuracy']}, precision={real_m['precision']}, recall={real_m['recall']}, f1={real_m['f1']}, auc={real_m['auc']}
- Train Synth -> Test Real: accuracy={synth_m['accuracy']}, precision={synth_m['precision']}, recall={synth_m['recall']}, f1={synth_m['f1']}, auc={synth_m['auc']}
- Delta (Synth - Real): accuracy={delta['accuracy']}, precision={delta['precision']}, recall={delta['recall']}, f1={delta['f1']}, auc={delta['auc']}

### Privacy
- Overall privacy score: **{priv['overall_privacy_score']}**
- Exact match rate: **{priv['exact_match_rate']}**
- Disclosure risk at distance < 0.06: **{priv['disclosure_risk_at_0_06']}**
- Nearest-neighbor distance (p01/p05/median): **{priv['nearest_neighbor_distance_p01']} / {priv['nearest_neighbor_distance_p05']} / {priv['nearest_neighbor_distance_median']}**
- Quasi-identifier uniqueness: **{priv['quasi_identifier_uniqueness']}**

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
"""


def run_pipeline(seed: int = 42) -> dict[str, Any]:
    real_rows = generate_medical_dataset(n_samples=2200, seed=seed)
    real_train, real_test = train_test_split(real_rows, test_ratio=0.2, seed=seed)

    synthesizer = ConditionalGaussianCopulaSynthesizer(
        target_column="disease",
        continuous_columns=CONTINUOUS_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        seed=seed,
    )
    synthesizer.fit(real_train)

    synthetic_rows = synthesizer.sample(n_samples=len(real_train))

    write_csv(real_rows, DATA_REAL_PATH)
    write_csv(synthetic_rows, DATA_SYNTH_PATH)

    util = utility_report(
        real_rows=real_train,
        synthetic_rows=synthetic_rows,
        continuous_columns=CONTINUOUS_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
    )

    downstream_real = evaluate_logistic_regression(real_train, real_test, seed=seed)
    downstream_synth = evaluate_logistic_regression(synthetic_rows, real_test, seed=seed)

    delta = {
        k: downstream_synth[k] - downstream_real[k]
        for k in downstream_real.keys()
    }

    priv = privacy_report(
        real_rows=real_train,
        synthetic_rows=synthetic_rows,
        continuous_columns=CONTINUOUS_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        seed=seed,
    )

    metrics: dict[str, Any] = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "real_total": len(real_rows),
            "real_train": len(real_train),
            "real_test": len(real_test),
            "synthetic_total": len(synthetic_rows),
        },
        "utility": util,
        "downstream": {
            "train_real_test_real": downstream_real,
            "train_synth_test_real": downstream_synth,
            "delta_synth_minus_real": _round_dict(delta),
        },
        "privacy": priv,
        "model": {
            "name": "Class-Conditional Gaussian Copula",
            "notes": "Pure Python implementation with separate per-class distributions.",
        },
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = _build_summary(metrics)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    report_md = _build_report_markdown(metrics)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    return metrics