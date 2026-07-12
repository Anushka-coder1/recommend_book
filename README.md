# Book Recommendation System

A portfolio-ready end-to-end machine learning project that recommends books similar to a user-selected title using collaborative filtering and a Streamlit web application.

## Project Overview

This project covers the full recommendation workflow:

- loading and validating the `Books.csv`, `Users.csv`, and `Ratings.csv` datasets
- cleaning missing values, duplicate rows, malformed years, and sparse interactions
- performing exploratory analysis to understand rating behavior and catalog coverage
- training a collaborative filtering recommender with cosine similarity and nearest-neighbor retrieval
- serving recommendations through a modern multi-page Streamlit application

## Features

- professional project structure with modular source files
- data cleaning and preprocessing pipeline
- popularity-based fallback recommendations
- item-item collaborative filtering for similar-book retrieval
- saved model artifacts in `models/`
- multi-page Streamlit UI with:
  - Popular Books page
  - Recommendation page
  - Dataset Statistics page
  - EDA Dashboard page
  - About Project page

## Dataset Information

The project uses the Book Recommendation dataset with:

- `Books.csv`
- `Users.csv`
- `Ratings.csv`

Expected location:

- primary: `data/`
- fallback: repository root

## Project Structure

```text
reccomend_book/
├── app.py
├── data/
├── models/
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── recommendation.py
│   └── utils.py
├── train_pipeline.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
python train_pipeline.py
streamlit run app.py
```

## Modeling Notes

- Collaborative filtering is used because ratings encode shared reader preferences.
- Cosine similarity is a strong fit for sparse user-book matrices because it compares direction rather than raw magnitude.
- Nearest-neighbor retrieval is simple, interpretable, and effective for serving similar books from an item-item similarity matrix.

## EDA Coverage

The notebook and dashboard cover:

- rating distribution
- top rated and most rated books
- most active authors and users
- publisher concentration
- publication year trends
- missing-value inspection
- pairwise numerical summaries where appropriate

## Screenshots

- `docs/screenshot-home.png` placeholder
- `docs/screenshot-recommendations.png` placeholder

## Future Improvements

- hybrid recommendations using book metadata and NLP features
- personalized user login state
- approximate nearest-neighbor serving for faster inference
- deployment to Streamlit Community Cloud or Docker
