"""Data loading and cleaning pipeline for the book recommender."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .utils import normalize_text, read_csv_with_fallback, resolve_data_path, safe_year


@dataclass(slots=True)
class DatasetBundle:
    """Container holding the raw or cleaned datasets."""

    books: pd.DataFrame
    users: pd.DataFrame
    ratings: pd.DataFrame


def load_datasets(data_dir: Path | None = None) -> DatasetBundle:
    """Load the three project datasets from disk."""
    base_dir = data_dir if data_dir else None
    books_path = (base_dir / "Books.csv") if base_dir else resolve_data_path("Books.csv")
    users_path = (base_dir / "Users.csv") if base_dir else resolve_data_path("Users.csv")
    ratings_path = (base_dir / "Ratings.csv") if base_dir else resolve_data_path("Ratings.csv")

    books = read_csv_with_fallback(books_path)
    users = read_csv_with_fallback(users_path)
    ratings = read_csv_with_fallback(ratings_path)
    return DatasetBundle(books=books, users=users, ratings=ratings)


def dataset_summary(frame: pd.DataFrame, name: str) -> dict[str, object]:
    """Produce a compact summary used in training logs and notebooks."""
    return {
        "dataset": name,
        "shape": frame.shape,
        "columns": frame.columns.tolist(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_values": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "sample_rows": frame.head(3).to_dict(orient="records"),
    }


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """Clean book metadata and keep only fields needed downstream."""
    cleaned = books.copy()
    # Removes spaces from column names.
    cleaned.columns = [column.strip() for column in cleaned.columns]
    cleaned["ISBN"] = cleaned["ISBN"].astype(str).str.strip()
    cleaned = cleaned[cleaned["ISBN"].ne("")]

    for column in ("Book-Title", "Book-Author", "Publisher"):
        cleaned[column] = cleaned[column].map(normalize_text)

    cleaned["Image-URL-L"] = cleaned["Image-URL-L"].fillna(cleaned["Image-URL-M"])
    cleaned["Image-URL-L"] = cleaned["Image-URL-L"].fillna(cleaned["Image-URL-S"])
    cleaned["Image-URL-L"] = cleaned["Image-URL-L"].map(normalize_text, na_action="ignore")

    cleaned["Year-Of-Publication"] = cleaned["Year-Of-Publication"].map(safe_year)
    median_year = int(cleaned["Year-Of-Publication"].dropna().median())
    cleaned["Year-Of-Publication"] = cleaned["Year-Of-Publication"].fillna(median_year).astype(int)

    cleaned = cleaned.drop_duplicates(subset=["ISBN"])
    return cleaned


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Clean user metadata without overfitting to age noise."""
    cleaned = users.copy()
    cleaned["Location"] = cleaned["Location"].map(normalize_text)
    cleaned["Age"] = pd.to_numeric(cleaned["Age"], errors="coerce")
    cleaned.loc[~cleaned["Age"].between(5, 100), "Age"] = pd.NA
    cleaned = cleaned.drop_duplicates(subset=["User-ID"])
    return cleaned


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Clean ratings and keep them within the valid explicit-rating range."""
    cleaned = ratings.copy()
    cleaned["ISBN"] = cleaned["ISBN"].astype(str).str.strip()
    cleaned["Book-Rating"] = pd.to_numeric(cleaned["Book-Rating"], errors="coerce")
    cleaned["User-ID"] = pd.to_numeric(cleaned["User-ID"], errors="coerce")

    cleaned = cleaned.dropna(subset=["ISBN", "Book-Rating", "User-ID"])
    cleaned["Book-Rating"] = cleaned["Book-Rating"].astype(int)
    cleaned["User-ID"] = cleaned["User-ID"].astype(int)
    cleaned = cleaned[cleaned["Book-Rating"].between(0, 10)]
    cleaned = cleaned.drop_duplicates()
    return cleaned


def clean_datasets(bundle: DatasetBundle) -> DatasetBundle:
    """Apply table-specific cleaning rules to the dataset bundle."""
    return DatasetBundle(
        books=clean_books(bundle.books),
        users=clean_users(bundle.users),
        ratings=clean_ratings(bundle.ratings),
    )


def build_modeling_frame(
    bundle: DatasetBundle,
    min_user_ratings: int = 20,
    min_book_ratings: int = 20,
    explicit_only: bool = True,
) -> pd.DataFrame:
    """Merge datasets and filter sparse interactions for collaborative filtering."""
    ratings = bundle.ratings.copy()
    if explicit_only:
        ratings = ratings[ratings["Book-Rating"] > 0]

    merged = ratings.merge(bundle.books, on="ISBN", how="inner")
    merged = merged.merge(bundle.users[["User-ID", "Location", "Age"]], on="User-ID", how="left")
    merged = merged.drop_duplicates(subset=["User-ID", "ISBN", "Book-Rating"])

    active_users = merged["User-ID"].value_counts()
    merged = merged[merged["User-ID"].isin(active_users[active_users >= min_user_ratings].index)]

    popular_titles = merged["Book-Title"].value_counts()
    merged = merged[merged["Book-Title"].isin(popular_titles[popular_titles >= min_book_ratings].index)]
    return merged.reset_index(drop=True)
