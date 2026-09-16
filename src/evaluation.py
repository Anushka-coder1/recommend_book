"""Offline evaluation utilities for the book recommender."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
@dataclass(slots=True)
class RecommenderMetrics:
    """Summary of evaluation metrics for one recommendation list."""
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    hit_rate: float
    reciprocal_rank: float
def precision_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Compute precision@k as the fraction of recommended items that are relevant."""
    recommended_list = list(recommended)[:k]
    relevant_set = set(relevant)
    if not recommended_list:
        return 0.0
    hits = len(set(recommended_list) & relevant_set)
    return hits / len(recommended_list)
def recall_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Compute recall@k against the true relevant set."""
    recommended_list = list(recommended)[:k]
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(set(recommended_list) & relevant_set)
    return hits / len(relevant_set)
def ndcg_at_k(recommended: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Compute discounted cumulative gain at k using binary relevance."""
    recommended_list = list(recommended)[:k]
    relevant_set = set(relevant)
    if not recommended_list or not relevant_set:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended_list, start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_relevant_count = min(len(relevant_set), k)
    ideal_dcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_relevant_count + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg
def reciprocal_rank(recommended: Iterable[str], relevant: Iterable[str]) -> float:
    """Compute MRR: reciprocal of the first relevant recommendation rank."""
    relevant_set = set(relevant)
    for rank, item in enumerate(recommended, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0
def summarize_metrics(recommended: Iterable[str], relevant: Iterable[str], k: int) -> RecommenderMetrics:
    """Bundle the key ranking metrics for an individual recommendation list."""
    recommended_list = list(recommended)
    return RecommenderMetrics(
        precision_at_k=precision_at_k(recommended_list, relevant, k),
        recall_at_k=recall_at_k(recommended_list, relevant, k),
        ndcg_at_k=ndcg_at_k(recommended_list, relevant, k),
        hit_rate=1.0 if set(recommended_list[:k]) & set(relevant) else 0.0,
        reciprocal_rank=reciprocal_rank(recommended_list, relevant),
    )
def rank_books_by_user_profile(
    similarity_matrix: np.ndarray,
    titles: list[str],
    user_ratings: dict[str, float],
) -> list[str]:
    """Score candidate books by their average similarity to a user's rated books."""
    title_to_index = {title: index for index, title in enumerate(titles)}
    candidates = [title for title in titles if title not in user_ratings]
    if not candidates:
        return []
    scores: list[tuple[str, float]] = []
    for candidate in candidates:
        candidate_index = title_to_index[candidate]
        candidate_scores = []
        for rated_title, rating in user_ratings.items():
            rated_index = title_to_index[rated_title]
            candidate_scores.append(similarity_matrix[rated_index, candidate_index] * float(rating))
        aggregate_score = float(np.mean(candidate_scores)) if candidate_scores else 0.0
        scores.append((candidate, aggregate_score))
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return [title for title, _ in ranked]
def evaluate_holdout_users(
    interactions: pd.DataFrame,
    similarity_matrix: np.ndarray,
    titles: list[str],
    k: int = 10,
    holdout_fraction: float = 0.2,
    min_history: int = 3,
    random_state: int = 42,
) -> dict[str, float]:
    """Evaluate a similarity-based recommender using per-user holdout splits."""
    rng = np.random.default_rng(random_state)
    records: list[RecommenderMetrics] = []
    for user_id, group in interactions.groupby("User-ID"):
        if len(group) < min_history + 1:
            continue
        sampled = group.sample(
            n=max(1, int(round(len(group) * holdout_fraction))),
            random_state=int(rng.integers(0, 10_000)),
        )
        test_titles = set(sampled["Book-Title"].tolist())
        train_group = group.drop(sampled.index)
        if train_group.empty or not test_titles:
            continue
        user_ratings = {
            row["Book-Title"]: float(row["Book-Rating"])
            for _, row in train_group[["Book-Title", "Book-Rating"]].drop_duplicates().iterrows()
        }
        if not user_ratings:
            continue
        ranked = rank_books_by_user_profile(similarity_matrix, titles, user_ratings)
        metrics = summarize_metrics(ranked, test_titles, k)
        records.append(metrics)
    if not records:
        return {
            "precision@k": 0.0,
            "recall@k": 0.0,
            "ndcg@k": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "users_evaluated": 0,
        }
    precision = float(np.mean([item.precision_at_k for item in records]))
    recall = float(np.mean([item.recall_at_k for item in records]))
    ndcg = float(np.mean([item.ndcg_at_k for item in records]))
    hit_rate = float(np.mean([item.hit_rate for item in records]))
    mrr = float(np.mean([item.reciprocal_rank for item in records]))
    return {
        "precision@k": precision,
        "recall@k": recall,
        "ndcg@k": ndcg,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "users_evaluated": len(records),
    }