"""Train and save the book recommendation artifacts."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
if VENDOR_DIR.exists():
    # Keep vendored packages as a fallback, but prefer the active environment.
    sys.path.append(str(VENDOR_DIR))

from src.data_preprocessing import build_modeling_frame, clean_datasets, dataset_summary, load_datasets
from src.feature_engineering import build_feature_artifacts
from src.utils import MODELS_DIR, ensure_directory


def main() -> None:
    """Run the full preprocessing and model-building pipeline."""
    ensure_directory(MODELS_DIR)

    raw_bundle = load_datasets()
    cleaned_bundle = clean_datasets(raw_bundle)

    for name, frame in (
        ("books", cleaned_bundle.books),
        ("users", cleaned_bundle.users),
        ("ratings", cleaned_bundle.ratings),
    ):
        summary = dataset_summary(frame, name)
        print(f"{summary['dataset']} shape: {summary['shape']}, duplicates: {summary['duplicate_rows']}")

    interactions = build_modeling_frame(
        cleaned_bundle,
        min_user_ratings=20,
        min_book_ratings=20,
        explicit_only=True,
    )
    print(f"Modeling interactions: {interactions.shape}")

    artifacts = build_feature_artifacts(interactions)

    serialized = {
        "popular_books.pkl": artifacts.popular_books,
        "pivot_table.pkl": artifacts.pivot_table,
        "similarity.pkl": artifacts.similarity_matrix,
        "book_lookup.pkl": artifacts.book_lookup,
    }

    for filename, artifact in serialized.items():
        with (MODELS_DIR / filename).open("wb") as file:
            pickle.dump(artifact, file)

    print("Saved model artifacts to models/.")


if __name__ == "__main__":
    sys.exit(main())
