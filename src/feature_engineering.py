"""Feature engineering and artifact creation for recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureArtifacts:
    """Artifacts required by the recommendation layer and Streamlit app."""

    popular_books: pd.DataFrame
    pivot_table: pd.DataFrame
    similarity_matrix: np.ndarray
    book_lookup: pd.DataFrame


def build_popular_books(
    interactions: pd.DataFrame,
    min_ratings: int = 250,
    limit: int = 50,
) -> pd.DataFrame:
    """Create a popularity-based fallback ranking for the homepage."""
    aggregates = (
        interactions.groupby("Book-Title")
        .agg(
            average_rating=("Book-Rating", "mean"),
            ratings_count=("Book-Rating", "count"),
        )
        .reset_index()
    )

    popular = aggregates[aggregates["ratings_count"] >= min_ratings].copy()
    popular = popular.merge(
        interactions[
            [
                "Book-Title",
                "Book-Author",
                "Publisher",
                "ISBN",
                "Image-URL-L",
                "Year-Of-Publication",
            ]
        ].drop_duplicates(subset=["Book-Title"]),
        on="Book-Title",
        how="left",
    )
    popular = popular.sort_values(
        by=["average_rating", "ratings_count"],
        ascending=[False, False],
    )
    popular = popular.drop_duplicates(subset=["Book-Title"]).head(limit).reset_index(drop=True)
    return popular


def build_user_book_matrix(interactions: pd.DataFrame) -> pd.DataFrame:
    """Build the pivot table used by collaborative filtering."""
    return interactions.pivot_table(
        index="Book-Title",
        columns="User-ID",
        values="Book-Rating",
        fill_value=0,
    )


def build_cosine_similarity_matrix(pivot_table: pd.DataFrame) -> np.ndarray:
    """Build an item-item cosine similarity matrix from the user-book pivot table."""
    matrix = pivot_table.to_numpy(dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


def build_book_lookup(interactions: pd.DataFrame) -> pd.DataFrame:
    """Keep one canonical metadata row per title for the app output."""
    ranking = (
        interactions.groupby("Book-Title")
        .agg(
            average_rating=("Book-Rating", "mean"),
            ratings_count=("Book-Rating", "count"),
        )
        .reset_index()
    )

    metadata = interactions[
        [
            "Book-Title",
            "Book-Author",
            "Publisher",
            "ISBN",
            "Image-URL-L",
            "Year-Of-Publication",
        ]
    ].drop_duplicates(subset=["Book-Title"])

    return ranking.merge(metadata, on="Book-Title", how="left")


def build_feature_artifacts(interactions: pd.DataFrame) -> FeatureArtifacts:
    """Generate all artifacts used for serving recommendations."""
    pivot_table = build_user_book_matrix(interactions)
    similarity_matrix = build_cosine_similarity_matrix(pivot_table)
    popular_books = build_popular_books(interactions)
    book_lookup = build_book_lookup(interactions)

    return FeatureArtifacts(
        popular_books=popular_books,
        pivot_table=pivot_table,
        similarity_matrix=similarity_matrix,
        book_lookup=book_lookup,
    )
