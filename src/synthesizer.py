"""Class-conditional Gaussian copula synthesizer for tabular medical data."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    if not vectors:
        return []
    n = len(vectors)
    d = len(vectors[0])
    means = [_mean([row[j] for row in vectors]) for j in range(d)]

    if n < 2:
        return [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]

    cov = [[0.0 for _ in range(d)] for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            s = 0.0
            for row in vectors:
                s += (row[i] - means[i]) * (row[j] - means[j])
            value = s / (n - 1)
            cov[i][j] = value
            cov[j][i] = value

    return cov


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                value = matrix[i][i] - s
                if value <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = math.sqrt(value)
            else:
                if L[j][j] == 0:
                    raise ValueError("Zero pivot during Cholesky")
                L[i][j] = (matrix[i][j] - s) / L[j][j]

    return L


def _make_pd(cov: list[list[float]], jitter_start: float = 1e-8, max_attempts: int = 8) -> list[list[float]]:
    n = len(cov)
    working = [row[:] for row in cov]
    jitter = jitter_start
    for _ in range(max_attempts):
        try:
            _ = _cholesky(working)
            return working
        except ValueError:
            for i in range(n):
                working[i][i] += jitter
            jitter *= 10.0
    return [[cov[i][j] + (1e-3 if i == j else 0.0) for j in range(n)] for i in range(n)]


def _matrix_vec_mul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    out = []
    for row in matrix:
        out.append(sum(a * b for a, b in zip(row, vector)))
    return out


def _rank_transform(values: list[float]) -> list[float]:
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])

    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _sample_from_probs(prob_map: dict[Any, float], rng: random.Random) -> Any:
    keys = list(prob_map.keys())
    probs = [prob_map[k] for k in keys]
    return rng.choices(keys, weights=probs, k=1)[0]


@dataclass
class ContinuousCopulaModel:
    columns: list[str]
    means: list[float]
    cov: list[list[float]]
    chol: list[list[float]]
    sorted_values: dict[str, list[float]]


class ConditionalGaussianCopulaSynthesizer:
    """Simple class-conditional copula synthesizer for tabular classification data."""

    def __init__(
        self,
        target_column: str,
        continuous_columns: list[str],
        categorical_columns: list[str],
        seed: int = 42,
    ) -> None:
        self.target_column = target_column
        self.continuous_columns = list(continuous_columns)
        self.categorical_columns = list(categorical_columns)
        self.seed = seed
        self._rng = random.Random(seed)

        self.class_probs: dict[Any, float] = {}
        self.class_models: dict[Any, ContinuousCopulaModel] = {}
        self.class_cat_probs: dict[Any, dict[str, dict[Any, float]]] = {}

    def fit(self, rows: list[dict[str, Any]]) -> None:
        by_class: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            y = row[self.target_column]
            by_class.setdefault(y, []).append(row)

        total = len(rows)
        self.class_probs = {y: len(group) / total for y, group in by_class.items()}

        for y, group in by_class.items():
            self.class_models[y] = self._fit_continuous_model(group)
            self.class_cat_probs[y] = self._fit_categorical_probs(group)

    def _fit_continuous_model(self, rows: list[dict[str, Any]]) -> ContinuousCopulaModel:
        normal = statistics.NormalDist()
        n = len(rows)

        sorted_values: dict[str, list[float]] = {
            col: sorted(float(row[col]) for row in rows) for col in self.continuous_columns
        }

        z_vectors = [[0.0 for _ in self.continuous_columns] for _ in range(n)]
        for j, col in enumerate(self.continuous_columns):
            values = [float(row[col]) for row in rows]
            ranks = _rank_transform(values)
            for i in range(n):
                u = (ranks[i] - 0.5) / n
                u = min(1.0 - 1e-10, max(1e-10, u))
                z_vectors[i][j] = normal.inv_cdf(u)

        means = [_mean([z[j] for z in z_vectors]) for j in range(len(self.continuous_columns))]
        cov = _covariance_matrix(z_vectors)
        cov = _make_pd(cov)
        chol = _cholesky(cov)

        return ContinuousCopulaModel(
            columns=self.continuous_columns,
            means=means,
            cov=cov,
            chol=chol,
            sorted_values=sorted_values,
        )

    def _fit_categorical_probs(self, rows: list[dict[str, Any]]) -> dict[str, dict[Any, float]]:
        out: dict[str, dict[Any, float]] = {}
        n = len(rows)

        for col in self.categorical_columns:
            counts: dict[Any, int] = {}
            for row in rows:
                key = row[col]
                counts[key] = counts.get(key, 0) + 1

            k = max(1, len(counts))
            probs = {key: (count + 1) / (n + k) for key, count in counts.items()}

            total = sum(probs.values())
            probs = {key: value / total for key, value in probs.items()}
            out[col] = probs

        return out

    def sample(self, n_samples: int) -> list[dict[str, Any]]:
        if not self.class_probs:
            raise RuntimeError("Synthesizer is not fitted")

        normal = statistics.NormalDist()
        class_values = list(self.class_probs.keys())
        class_weights = [self.class_probs[c] for c in class_values]

        rows: list[dict[str, Any]] = []
        for _ in range(n_samples):
            y = self._rng.choices(class_values, weights=class_weights, k=1)[0]
            model = self.class_models[y]
            cat_probs = self.class_cat_probs[y]

            eps = [self._rng.gauss(0.0, 1.0) for _ in model.columns]
            z = _matrix_vec_mul(model.chol, eps)
            z = [zj + muj for zj, muj in zip(z, model.means)]

            row: dict[str, Any] = {}
            for j, col in enumerate(model.columns):
                u = normal.cdf(z[j])
                u = min(1.0 - 1e-10, max(1e-10, u))
                sorted_vals = model.sorted_values[col]
                idx = int(math.floor(u * len(sorted_vals)))
                idx = min(len(sorted_vals) - 1, max(0, idx))
                value = float(sorted_vals[idx])

                min_v = sorted_vals[0]
                max_v = sorted_vals[-1]

                if col == "age":
                    value = _clip(value + self._rng.gauss(0.0, 3.5), min_v, max_v)
                    row[col] = int(round(value))
                else:
                    noise_std = max(0.9, 0.06 * (max_v - min_v))
                    value = _clip(value + self._rng.gauss(0.0, noise_std), min_v, max_v)
                    row[col] = round(value, 2)

            for col in self.categorical_columns:
                sampled = _sample_from_probs(cat_probs[col], self._rng)
                if col in {"smoker", "family_history", "exercise_level", "alcohol_use"}:
                    sampled = int(sampled)
                row[col] = sampled

            row[self.target_column] = int(y)
            rows.append(row)

        return rows
