"""Streamlit interface for the book recommendation system."""

from __future__ import annotations

import sys
from pathlib import Path

# Fallback mechanism to load packages from the local "_vendor"
# directory if they are unavailable in the current Python environment.

VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
try:
    import pandas as pd
    import streamlit as st
except ImportError:
    if VENDOR_DIR.exists():
        sys.path.insert(0, str(VENDOR_DIR))
        import pandas as pd
        import streamlit as st
    else:
        raise

from src.data_preprocessing import build_modeling_frame, clean_datasets, load_datasets
from src.recommendation import BookRecommender


st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(214, 163, 74, 0.18), transparent 28%),
            linear-gradient(135deg, #f7f1e3 0%, #fefaf3 48%, #e9f0f5 100%);
        color: #1f2933;
    }
    .hero-card, .book-card, .metric-card {
        background: rgba(255, 255, 255, 0.80);
        border: 1px solid rgba(125, 97, 54, 0.12);
        border-radius: 18px;
        padding: 1.1rem;
        box-shadow: 0 18px 40px rgba(56, 43, 24, 0.08);
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #3b2f2f;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #5c6773;
        font-size: 1rem;
        line-height: 1.7;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #3b2f2f;
        margin: 0.4rem 0 1rem 0;
    }
    .book-meta {
        font-size: 0.95rem;
        line-height: 1.55;
        color: #52606d;
    }
    .footer {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        color: #52606d;
        font-size: 0.9rem;
    }
</style>
"""

@st.cache_resource(show_spinner=False)
def load_recommender() -> BookRecommender:
    """Load the persisted recommendation artifacts once per session."""
    try:
        return BookRecommender()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Model artifacts are missing. Run `python train_pipeline.py` before starting the app."
        ) from exc


@st.cache_data(show_spinner=False)
def load_statistics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cleaned datasets for dashboard summaries."""
    cleaned_bundle = clean_datasets(load_datasets())
    interactions = build_modeling_frame(cleaned_bundle)
    return cleaned_bundle.books, cleaned_bundle.users, interactions


def build_project_metrics(
    books: pd.DataFrame,
    users: pd.DataFrame,
    interactions: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Create reusable high-level project metrics for the UI."""
    return [
        ("Books", f"{books['Book-Title'].nunique():,}"),
        ("Users", f"{users['User-ID'].nunique():,}"),
        ("Filtered Ratings", f"{len(interactions):,}"),
        ("Authors", f"{books['Book-Author'].nunique():,}"),
    ]


def render_sidebar(books: pd.DataFrame, users: pd.DataFrame, interactions: pd.DataFrame) -> str:
    """Render the sidebar and return the selected page."""
    st.sidebar.title("Book Recommender")
    st.sidebar.caption("Portfolio-ready collaborative filtering project")
    st.sidebar.markdown("---")
    st.sidebar.write(
        "Use the pages below to browse popular titles, generate similar-book recommendations, "
        "and inspect the dataset that powers the model."
    )

    for label, value in build_project_metrics(books, users, interactions):
        st.sidebar.metric(label, value)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Tech Stack**

        `Python` `Pandas` `NumPy` `Streamlit` `Collaborative Filtering`
        """
    )

    st.sidebar.markdown("---")
    return st.sidebar.radio(
        "Choose a page",
        (
            "Popular Books",
            "Recommendation",
            "Dataset Statistics",
            "EDA Dashboard",
            "About Project",
        ),
    )


def render_book_card(book: dict[str, object]) -> None:
    """Render one book recommendation card."""
    with st.container():
        st.markdown("<div class='book-card'>", unsafe_allow_html=True)
        image_value = str(book["image"]).strip()
        if image_value.startswith(("http://", "https://")):
            st.image(image_value, use_container_width=True)
        else:
            st.markdown("### No cover available")
        st.markdown(f"**{book['title']}**")
        st.markdown(
            (
                "<div class='book-meta'>"
                f"Author: {book['author']}<br>"
                f"Average Rating: {book['rating']}<br>"
                f"Ratings Count: {book['ratings_count']}<br>"
                f"ISBN: {book['isbn']}<br>"
                f"Publisher: {book['publisher']}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_homepage(recommender: BookRecommender) -> None:
    """Show the landing section and popular books."""
    books, users, interactions = load_statistics()
    metrics = build_project_metrics(books, users, interactions)

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Book Recommendation System</div>
            <div class="hero-subtitle">
                Collaborative filtering meets a portfolio-ready interface. Explore popular books,
                inspect dataset scale, and discover titles similar to the ones you already love.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(len(metrics))
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Popular Books</div>", unsafe_allow_html=True)
    popular = recommender.popular_catalog(limit=12)
    columns = st.columns(4)
    for idx, (_, row) in enumerate(popular.iterrows()):
        with columns[idx % 4]:
            render_book_card(
                {
                    "title": row["Book-Title"],
                    "author": row["Book-Author"],
                    "image": row["Image-URL-L"],
                    "rating": round(float(row["average_rating"]), 2),
                    "ratings_count": int(row["ratings_count"]),
                    "isbn": row["ISBN"],
                    "publisher": row["Publisher"],
                }
            )


def render_recommendation_page(recommender: BookRecommender) -> None:
    """Let the user choose a seed title and receive similar books."""
    st.markdown("<div class='section-title'>Find Similar Books</div>", unsafe_allow_html=True)
    st.write(
        "Search for a title from the filtered catalog and generate nearby books based on "
        "reader behavior patterns."
    )
    selected_title = st.selectbox(
        "Search and select a book",
        options=recommender.available_titles(),
        index=None,
        placeholder="Start typing a book title...",
    )

    if st.button("Recommend", type="primary", use_container_width=True):
        if not selected_title:
            st.error("Select a book title before requesting recommendations.")
            return

        with st.spinner("Generating recommendations..."):
            try:
                recommendations = recommender.recommend(selected_title)
            except ValueError as exc:
                st.error(str(exc))
                return

        st.success(f"Showing books similar to '{selected_title}'.")
        columns = st.columns(min(5, len(recommendations)))
        for idx, recommendation in enumerate(recommendations):
            with columns[idx % len(columns)]:
                render_book_card(
                    {
                        "title": recommendation.title,
                        "author": recommendation.author,
                        "image": recommendation.cover_image,
                        "rating": recommendation.average_rating,
                        "ratings_count": recommendation.ratings_count,
                        "isbn": recommendation.isbn,
                        "publisher": recommendation.publisher,
                    }
                )


def render_statistics_page() -> None:
    """Show headline dataset statistics and table previews."""
    books, users, interactions = load_statistics()

    st.markdown("<div class='section-title'>Dataset Statistics</div>", unsafe_allow_html=True)
    metrics = st.columns(4)
    metric_values = build_project_metrics(books, users, interactions)

    for column, (label, value) in zip(metrics, metric_values):
        with column:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Sample of Filtered Modeling Data")
    st.dataframe(
        interactions[
            ["Book-Title", "Book-Author", "User-ID", "Book-Rating", "Publisher", "Year-Of-Publication"]
        ].head(20),
        use_container_width=True,
    )


def render_eda_page() -> None:
    """Render lightweight EDA directly in Streamlit for quick exploration."""
    books, users, interactions = load_statistics()

    st.markdown("<div class='section-title'>EDA Dashboard</div>", unsafe_allow_html=True)
    st.write("A quick visual summary of rating behavior, author activity, publisher concentration, and missing values.")
    rating_distribution = interactions["Book-Rating"].value_counts().sort_index()
    st.subheader("Rating Distribution")
    st.bar_chart(rating_distribution)

    top_books = (
        interactions.groupby("Book-Title")["Book-Rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    st.subheader("Top 10 Most Rated Books")
    st.bar_chart(top_books)

    top_authors = (
        interactions.groupby("Book-Author")["Book-Rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    st.subheader("Most Active Authors")
    st.bar_chart(top_authors)

    publisher_counts = books["Publisher"].value_counts().head(10)
    st.subheader("Top Publishers")
    st.bar_chart(publisher_counts)

    missing_summary = pd.DataFrame(
        {
            "books": books.isna().sum(),
            "users": users.reindex(columns=["User-ID", "Location", "Age"]).isna().sum(),
        }
    ).fillna(0)
    st.subheader("Missing Values Overview")
    st.dataframe(missing_summary, use_container_width=True)


def render_about_page() -> None:
    """Explain the project and modeling choices."""
    st.markdown("<div class='section-title'>About This Project</div>", unsafe_allow_html=True)
    st.write(
        """
        This project uses collaborative filtering with cosine similarity to recommend books
        based on shared user behavior. Sparse user-book interactions are filtered to retain
        active readers and sufficiently rated titles, which improves recommendation quality
        and reduces noise.
        """
    )
    st.write(
        """
        Why this approach:
        collaborative filtering captures latent taste patterns without needing book text,
        cosine similarity works well on sparse rating vectors, and a popularity-based catalog
        provides a practical fallback for cold-start browsing.
        """
    )
    st.info(
        "Performance notes: model artifacts are cached, dataset summaries are memoized, and "
        "recommendations are served from a precomputed similarity matrix."
    )


def main() -> None:
    """Run the Streamlit application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    try:
        recommender = load_recommender()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    books, users, interactions = load_statistics()
    page = render_sidebar(books, users, interactions)

    if page == "Popular Books":
        render_homepage(recommender)
    elif page == "Recommendation":
        render_recommendation_page(recommender)
    elif page == "Dataset Statistics":
        render_statistics_page()
    elif page == "EDA Dashboard":
        render_eda_page()
    else:
        render_about_page()

    st.markdown(
        "<div class='footer'>Built with Python, collaborative filtering, and Streamlit.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
