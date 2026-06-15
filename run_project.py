"""Run script for the synthetic medical data project."""

from __future__ import annotations

from src.pipeline import run_pipeline


if __name__ == "__main__":
    metrics = run_pipeline(seed=42)

    print("Project pipeline completed successfully.")
    print("Real rows:", metrics["data"]["real_total"])
    print("Synthetic rows:", metrics["data"]["synthetic_total"])
    print("Utility score:", metrics["utility"]["overall_utility_score"])
    print("Privacy score:", metrics["privacy"]["overall_privacy_score"])
    print("Results written to results/ and report/ directories.")