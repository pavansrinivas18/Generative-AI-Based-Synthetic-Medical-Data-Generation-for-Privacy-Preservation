"""Utility and privacy metrics for synthetic medical datasets."""

from __future__ import annotations

import math
import random
from typing import Any


def _empirical_cdf(values: list[float], x: float) -> float:
    if not values:
        return 0.0
    count = 0
    for v in values:
        if v <= x:
            count += 1
    return count / len(values)


def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    points = sorted(set(sample_a + sample_b))
    if not points:
        return 0.0
    max_diff = 0.0
    for x in points:
        diff = abs(_empirical_cdf(sample_a, x) - _empirical_cdf(sample_b, x))
        if diff > max_diff:
            max_diff = diff
    return max_diff


def _value_counts(values: list[Any]) -> dict[Any, int]:
    counts: dict[Any, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def _categorical_similarity(real_values: list[Any], synth_values: list[Any]) -> float:
    real_counts = _value_counts(real_values)
    synth_counts = _value_counts(synth_values)
    keys = set(real_counts) | set(synth_counts)

    real_n = len(real_values)
    synth_n = len(synth_values)

    tvd = 0.0
    for key in keys:
        p = real_counts.get(key, 0) / real_n
        q = synth_counts.get(key, 0) / synth_n
        tvd += abs(p - q)
    tvd *= 0.5

    return max(0.0, 1.0 - tvd)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float], mean_value: float) -> float:
    if len(values) < 2:
        return 1.0
    variance = sum((x - mean_value) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) if variance > 1e-12 else 1.0


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = _mean(x)
    my = _mean(y)
    sx = _std(x, mx)
    sy = _std(y, my)

    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - 1)
    return cov / (sx * sy) if sx > 0 and sy > 0 else 0.0


def correlation_matrix(rows: list[dict[str, Any]], columns: list[str]) -> list[list[float]]:
    data = {col: [float(row[col]) for row in rows] for col in columns}
    matrix = []
    for c1 in columns:
        matrix.append([_corr(data[c1], data[c2]) for c2 in columns])
    return matrix


def correlation_difference_score(
    real_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    columns: list[str],
) -> float:
    real_corr = correlation_matrix(real_rows, columns)
    synth_corr = correlation_matrix(synthetic_rows, columns)

    diffs = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            diffs.append(abs(real_corr[i][j] - synth_corr[i][j]))

    if not diffs:
        return 1.0

    avg_diff = sum(diffs) / len(diffs)
    return max(0.0, 1.0 - avg_diff / 2.0)


def utility_report(
    real_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    continuous_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Any]:
    per_feature: dict[str, dict[str, float]] = {}

    continuous_scores = []
    for col in continuous_columns:
        real_vals = [float(r[col]) for r in real_rows]
        synth_vals = [float(r[col]) for r in synthetic_rows]
        ks = ks_statistic(real_vals, synth_vals)
        score = max(0.0, 1.0 - ks)
        per_feature[col] = {
            "ks": round(ks, 4),
            "similarity": round(score, 4),
        }
        continuous_scores.append(score)

    categorical_scores = []
    for col in categorical_columns:
        real_vals = [r[col] for r in real_rows]
        synth_vals = [r[col] for r in synthetic_rows]
        sim = _categorical_similarity(real_vals, synth_vals)
        per_feature[col] = {
            "similarity": round(sim, 4),
        }
        categorical_scores.append(sim)

    corr_cols = list(continuous_columns)
    corr_score = correlation_difference_score(real_rows, synthetic_rows, corr_cols)

    overall = (
        0.5 * (sum(continuous_scores) / max(1, len(continuous_scores)))
        + 0.2 * (sum(categorical_scores) / max(1, len(categorical_scores)))
        + 0.3 * corr_score
    )

    return {
        "per_feature": per_feature,
        "continuous_similarity": round(sum(continuous_scores) / max(1, len(continuous_scores)), 4),
        "categorical_similarity": round(sum(categorical_scores) / max(1, len(categorical_scores)), 4),
        "correlation_similarity": round(corr_score, 4),
        "overall_utility_score": round(overall, 4),
    }


def _row_signature(row: dict[str, Any], columns: list[str]) -> str:
    parts = []
    for col in columns:
        value = row[col]
        if isinstance(value, float):
            parts.append(f"{value:.2f}")
        else:
            parts.append(str(value))
    return "|".join(parts)


def _min_max(values: list[float]) -> tuple[float, float]:
    return min(values), max(values)


def _row_distance(
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    continuous_columns: list[str],
    categorical_columns: list[str],
    ranges: dict[str, tuple[float, float]],
) -> float:
    sq_sum = 0.0
    dim = 0

    for col in continuous_columns:
        min_v, max_v = ranges[col]
        denom = max(max_v - min_v, 1e-6)
        da = float(row_a[col])
        db = float(row_b[col])
        diff = (da - db) / denom
        sq_sum += diff * diff
        dim += 1

    for col in categorical_columns:
        mismatch = 0.0 if row_a[col] == row_b[col] else 1.0
        sq_sum += mismatch
        dim += 1

    return math.sqrt(sq_sum / max(1, dim))


def privacy_report(
    real_rows: list[dict[str, Any]],
    synthetic_rows: list[dict[str, Any]],
    continuous_columns: list[str],
    categorical_columns: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    all_cols = continuous_columns + categorical_columns + ["disease"]

    real_signatures = {_row_signature(row, all_cols) for row in real_rows}
    synthetic_signatures = [_row_signature(row, all_cols) for row in synthetic_rows]

    exact_matches = sum(1 for sig in synthetic_signatures if sig in real_signatures)
    exact_match_rate = exact_matches / max(1, len(synthetic_rows))

    ranges = {
        col: _min_max([float(r[col]) for r in real_rows])
        for col in continuous_columns
    }

    rng = random.Random(seed)
    ref_real = list(real_rows)
    if len(ref_real) > 1200:
        ref_real = rng.sample(ref_real, 1200)

    min_distances = []
    for srow in synthetic_rows:
        best = float("inf")
        for rrow in ref_real:
            d = _row_distance(srow, rrow, continuous_columns, categorical_columns, ranges)
            if d < best:
                best = d
        min_distances.append(best)

    sorted_d = sorted(min_distances)

    def percentile(p: float) -> float:
        if not sorted_d:
            return 0.0
        idx = int(round((len(sorted_d) - 1) * p))
        idx = max(0, min(len(sorted_d) - 1, idx))
        return sorted_d[idx]

    disclosure_threshold = 0.06
    disclosure_risk = sum(1 for d in min_distances if d < disclosure_threshold) / max(1, len(min_distances))

    quasi_cols = ["age", "gender", "bmi", "glucose", "systolic_bp"]
    quasi_signatures = [_row_signature(row, quasi_cols) for row in synthetic_rows]
    unique_quasi = len(set(quasi_signatures)) / max(1, len(quasi_signatures))

    privacy_score = 1.0
    privacy_score -= min(1.0, exact_match_rate * 25.0)
    privacy_score -= min(1.0, disclosure_risk * 4.0) * 0.7
    privacy_score -= (1.0 - unique_quasi) * 0.3
    privacy_score = max(0.0, privacy_score)

    return {
        "exact_match_rate": round(exact_match_rate, 6),
        "nearest_neighbor_distance_p01": round(percentile(0.01), 4),
        "nearest_neighbor_distance_p05": round(percentile(0.05), 4),
        "nearest_neighbor_distance_median": round(percentile(0.5), 4),
        "disclosure_risk_at_0_06": round(disclosure_risk, 4),
        "quasi_identifier_uniqueness": round(unique_quasi, 4),
        "overall_privacy_score": round(privacy_score, 4),
    }