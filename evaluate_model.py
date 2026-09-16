"""Run offline evaluation for the collaborative-filtering recommendation model."""

from __future__ import annotations

import argparse

from src.data_preprocessing import build_modeling_frame, clean_datasets, load_datasets
from src.evaluation import evaluate_holdout_users
from src.feature_engineering import build_feature_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the book recommender with holdout validation.")
    parser.add_argument("--top-k", type=int, default=10, help="Cutoff for top-k metrics")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="Fraction of each user's ratings to reserve for testing")
    parser.add_argument("--min-history", type=int, default=3, help="Minimum number of training ratings per user to evaluate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = clean_datasets(load_datasets())
    interactions = build_modeling_frame(bundle, min_user_ratings=20, min_book_ratings=20, explicit_only=True)
    artifacts = build_feature_artifacts(interactions)

    metrics = evaluate_holdout_users(
        interactions=interactions,
        similarity_matrix=artifacts.similarity_matrix,
        titles=artifacts.pivot_table.index.tolist(),
        k=args.top_k,
        holdout_fraction=args.holdout_fraction,
        min_history=args.min_history,
    )

    print("Offline evaluation summary")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()