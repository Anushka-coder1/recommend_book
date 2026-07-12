"""Recommendation utilities and serving wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .utils import MODELS_DIR


@dataclass(slots=True)
class RecommendationResult:
    """Single recommendation payload displayed by the app."""

    title: str
    author: str
    cover_image: str
    average_rating: float
    ratings_count: int
    isbn: str
    publisher: str
    year_of_publication: int
    similarity_score: float


class BookRecommender:
    """Load persisted artifacts and serve similar-book recommendations."""

    def __init__(self, models_dir: Path | None = None) -> None:
        base_dir = models_dir or MODELS_DIR
        self.popular_books = self._load_pickle(base_dir / "popular_books.pkl")
        self.pivot_table = self._load_pickle(base_dir / "pivot_table.pkl")
        self.similarity_matrix = self._load_pickle(base_dir / "similarity.pkl")
        self.book_lookup = self._load_pickle(base_dir / "book_lookup.pkl")
        self._titles = self.pivot_table.index.tolist()
        self._title_to_index = {title: index for index, title in enumerate(self._titles)}
        self._book_lookup_by_title = self.book_lookup.set_index("Book-Title", drop=False)

    @staticmethod
    def _load_pickle(path: Path) -> object:
        with path.open("rb") as file:
            return pickle.load(file)

    def available_titles(self) -> list[str]:
        """Return alphabetically sorted book titles for the UI."""
        return sorted(self._titles)

    def recommend(self, book_name: str, top_n: int = 5) -> list[RecommendationResult]:
        """Return the most similar books to the selected title."""
        if book_name not in self._title_to_index:
            raise ValueError(f"'{book_name}' is not available in the recommendation catalog.")

        index = self._title_to_index[book_name]
        similarity_scores = self.similarity_matrix[index]
        candidate_count = min(top_n + 1, similarity_scores.shape[0])
        top_indices = np.argpartition(similarity_scores, -candidate_count)[-candidate_count:]
        ranked_indices = top_indices[np.argsort(similarity_scores[top_indices])[::-1]]

        recommendations: list[RecommendationResult] = []
        for similar_index in ranked_indices:
            if similar_index == index:
                continue

            similar_index = int(similar_index)
            score = similarity_scores[similar_index]
            title = self._titles[similar_index]
            details = self._book_lookup_by_title.loc[title]
            recommendations.append(
                RecommendationResult(
                    title=title,
                    author=str(details["Book-Author"]),
                    cover_image=str(details["Image-URL-L"]),
                    average_rating=round(float(details["average_rating"]), 2),
                    ratings_count=int(details["ratings_count"]),
                    isbn=str(details["ISBN"]),
                    publisher=str(details["Publisher"]),
                    year_of_publication=int(details["Year-Of-Publication"]),
                    similarity_score=round(float(score), 4),
                )
            )
            if len(recommendations) == top_n:
                break

        return recommendations

    def popular_catalog(self, limit: int = 12) -> pd.DataFrame:
        """Return a homepage-friendly slice of popular books."""
        return self.popular_books.head(limit).copy()
