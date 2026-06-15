"""Dataset generation and CSV helpers for the medical synthetic data project."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Any

FEATURE_COLUMNS = [
    "age",
    "gender",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "cholesterol",
    "smoker",
    "family_history",
    "exercise_level",
    "alcohol_use",
]

TARGET_COLUMN = "disease"

CONTINUOUS_COLUMNS = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "cholesterol",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "smoker",
    "family_history",
    "exercise_level",
    "alcohol_use",
]

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generate_medical_dataset(n_samples: int = 2000, seed: int = 42) -> list[dict[str, Any]]:
    """Generate a realistic non-identifying medical tabular dataset."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for _ in range(n_samples):
        age = int(round(_clip(rng.gauss(49, 16), 18, 90)))
        gender = "M" if rng.random() < 0.52 else "F"

        bmi_base = 24.5 + (age - 45) * 0.03 + (1.0 if gender == "M" else 0.0)
        bmi = round(_clip(rng.gauss(bmi_base, 4.2), 16.0, 45.0), 2)

        systolic_bp = round(_clip(rng.gauss(114 + (age - 35) * 0.58 + (bmi - 25) * 0.7, 11), 85, 210), 1)
        diastolic_bp = round(_clip(rng.gauss(74 + (age - 35) * 0.28 + (bmi - 25) * 0.35, 8), 55, 130), 1)

        smoker_prob = _clip(0.16 + (0.05 if age < 35 else 0.0) + (0.04 if gender == "M" else 0.0), 0.05, 0.42)
        smoker = 1 if rng.random() < smoker_prob else 0

        family_history = 1 if rng.random() < 0.34 else 0

        exercise_level = rng.choices([0, 1, 2], weights=[0.29, 0.49, 0.22], k=1)[0]
        alcohol_use = rng.choices([0, 1, 2], weights=[0.41, 0.46, 0.13], k=1)[0]

        glucose_mu = 90 + (age - 40) * 0.42 + (bmi - 25) * 1.2 + smoker * 6.0 - exercise_level * 3.2
        glucose = round(_clip(rng.gauss(glucose_mu, 13.5), 60, 280), 1)

        chol_mu = 168 + (age - 40) * 0.55 + (bmi - 25) * 1.0 + smoker * 5.8 + family_history * 6.0
        cholesterol = round(_clip(rng.gauss(chol_mu, 21.0), 110, 390), 1)

        logit = (
            -3.8
            + 0.051 * age
            + 0.08 * (bmi - 25)
            + 0.021 * (systolic_bp - 120)
            + 0.024 * (glucose - 100)
            + 0.014 * (cholesterol - 180)
            + 0.74 * smoker
            + 0.88 * family_history
            - 0.34 * exercise_level
            + 0.22 * alcohol_use
            + (0.15 if gender == "M" else 0.0)
            + rng.gauss(0.0, 0.38)
        )
        disease_prob = _clip(_sigmoid(logit), 0.003, 0.997)
        disease = 1 if rng.random() < disease_prob else 0

        rows.append(
            {
                "age": age,
                "gender": gender,
                "bmi": bmi,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "glucose": glucose,
                "cholesterol": cholesterol,
                "smoker": smoker,
                "family_history": family_history,
                "exercise_level": exercise_level,
                "alcohol_use": alcohol_use,
                "disease": disease,
            }
        )

    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _convert_value(column: str, value: str) -> Any:
    if column == "gender":
        return value
    if column in {"age", "smoker", "family_history", "exercise_level", "alcohol_use", "disease"}:
        return int(value)
    return float(value)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            casted = {col: _convert_value(col, row[col]) for col in ALL_COLUMNS}
            rows.append(casted)
    return rows


def train_test_split(rows: list[dict[str, Any]], test_ratio: float = 0.2, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    split_idx = int(round(len(shuffled) * (1.0 - test_ratio)))
    split_idx = max(1, min(split_idx, len(shuffled) - 1))
    return shuffled[:split_idx], shuffled[split_idx:]
