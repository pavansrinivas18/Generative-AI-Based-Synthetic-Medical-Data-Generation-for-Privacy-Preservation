"""Downstream binary classification model and metrics (pure Python)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


ENCODED_FEATURES = [
    "age",
    "gender_M",
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


def _sigmoid(x: float) -> float:
    if x < -30:
        return 1e-13
    if x > 30:
        return 1.0 - 1e-13
    return 1.0 / (1.0 + math.exp(-x))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], mean_value: float) -> float:
    if not values:
        return 1.0
    variance = sum((x - mean_value) ** 2 for x in values) / max(1, len(values) - 1)
    return math.sqrt(variance) if variance > 1e-12 else 1.0


def _encode_row(row: dict[str, Any]) -> list[float]:
    return [
        float(row["age"]),
        1.0 if row["gender"] == "M" else 0.0,
        float(row["bmi"]),
        float(row["systolic_bp"]),
        float(row["diastolic_bp"]),
        float(row["glucose"]),
        float(row["cholesterol"]),
        float(row["smoker"]),
        float(row["family_history"]),
        float(row["exercise_level"]),
        float(row["alcohol_use"]),
    ]


def rows_to_xy(rows: list[dict[str, Any]]) -> tuple[list[list[float]], list[int]]:
    X = [_encode_row(row) for row in rows]
    y = [int(row["disease"]) for row in rows]
    return X, y


@dataclass
class StandardScaler:
    means: list[float]
    stds: list[float]

    @classmethod
    def fit(cls, X: list[list[float]]) -> "StandardScaler":
        if not X:
            raise ValueError("Cannot fit scaler on empty dataset")

        n_features = len(X[0])
        columns = [[row[j] for row in X] for j in range(n_features)]

        means = [_mean(col) for col in columns]
        stds = [_std(col, mu) for col, mu in zip(columns, means)]
        return cls(means=means, stds=stds)

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        return [
            [
                (value - self.means[j]) / self.stds[j]
                for j, value in enumerate(row)
            ]
            for row in X
        ]


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.05
    n_epochs: int = 600
    l2_lambda: float = 1e-3
    seed: int = 42

    def __post_init__(self) -> None:
        self.weights: list[float] = []
        self.bias: float = 0.0

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        if not X:
            raise ValueError("Training data is empty")

        n_samples = len(X)
        n_features = len(X[0])

        rng = random.Random(self.seed)
        self.weights = [rng.uniform(-0.01, 0.01) for _ in range(n_features)]
        self.bias = 0.0

        for _ in range(self.n_epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0

            for row, target in zip(X, y):
                linear = _dot(self.weights, row) + self.bias
                pred = _sigmoid(linear)
                error = pred - target

                for j in range(n_features):
                    grad_w[j] += error * row[j]
                grad_b += error

            for j in range(n_features):
                grad_w[j] = (grad_w[j] / n_samples) + self.l2_lambda * self.weights[j]
                self.weights[j] -= self.learning_rate * grad_w[j]

            grad_b /= n_samples
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [_sigmoid(_dot(self.weights, row) + self.bias) for row in X]

    def predict(self, X: list[list[float]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def roc_auc_score(y_true: list[int], y_score: list[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        pos_in_tie = sum(1 for k in range(i, j) if pairs[k][1] == 1)
        rank_sum_pos += avg_rank * pos_in_tie
        i = j

    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return auc


def evaluate_logistic_regression(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, float]:
    X_train, y_train = rows_to_xy(train_rows)
    X_test, y_test = rows_to_xy(test_rows)

    scaler = StandardScaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegressionGD(seed=seed)
    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)
    preds = [1 if p >= 0.5 else 0 for p in probs]

    precision, recall, f1 = precision_recall_f1(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
    }