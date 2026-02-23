import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_title_index(movies):
    """
    Safely map title -> first matching row index.
    Handles duplicate titles.
    """
    title_index = (
        movies.reset_index()
        .groupby("title")["index"]
        .first()
    )
    return title_index


def build_movieid_to_index(movies: pd.DataFrame) -> dict:
    # movieId -> row index
    if "movieId" not in movies.columns:
        raise ValueError("movies must contain a 'movieId' column.")
    return dict(zip(movies["movieId"].values, movies.index.values))


def recommend_similar_movie(
    title: str,
    movies: pd.DataFrame,
    similarity_matrix: np.ndarray,
    title_index: pd.Series,
    top_n: int = 10
) -> pd.DataFrame:
    idx = int(title_index[title])
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # skip itself
    sim_scores = sim_scores[1: top_n + 1]

    movie_indices = [i[0] for i in sim_scores]
    similarities = [i[1] for i in sim_scores]

    out = movies.iloc[movie_indices].copy()
    out["similarity_score"] = similarities
    return out


def build_user_profile_vector(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix,
    movieid_to_index: dict,
    min_rating: float = 4.0
):
    """
    Build a user "taste" vector by averaging TF-IDF vectors of movies the user rated highly.
    Weighted by rating.
    """
    if "userId" not in ratings.columns or "movieId" not in ratings.columns or "rating" not in ratings.columns:
        raise ValueError("ratings.csv must contain columns: userId, movieId, rating")

    user_rows = ratings[ratings["userId"] == user_id].copy()
    if user_rows.empty:
        return None, []

    liked = user_rows[user_rows["rating"] >= min_rating].copy()
    if liked.empty:
        # fallback: take top-N rated
        liked = user_rows.sort_values("rating", ascending=False).head(10)

    # keep only movies that exist in our movies table
    liked["row_idx"] = liked["movieId"].map(movieid_to_index)
    liked = liked.dropna(subset=["row_idx"])
    if liked.empty:
        return None, []

    idxs = liked["row_idx"].astype(int).values
    weights = liked["rating"].values.astype(float)

    # Weighted average of sparse/dense rows
    # tfidf_matrix[idxs] returns a matrix of rows
    user_mat = tfidf_matrix[idxs]
    # multiply each row by its weight then average
    # works for sparse matrices
    weighted = user_mat.multiply(weights[:, None]) if hasattr(user_mat, "multiply") else (user_mat * weights[:, None])
    user_vec = weighted.mean(axis=0)

    # Convert to 1D dense vector for cosine similarity calls
    if hasattr(user_vec, "A1"):
        user_vec = user_vec.A1
    else:
        user_vec = np.asarray(user_vec).ravel()

    # Return also the titles of top-liked movies for explanations
    liked = liked.sort_values("rating", ascending=False)
    liked_titles = movies.loc[liked["row_idx"].astype(int), "title"].head(5).tolist()
    return user_vec, liked_titles


def recommend_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    tfidf_matrix,
    movieid_to_index: dict,
    top_n: int = 10,
    min_rating: float = 4.0,
    exclude_rated: bool = True
) -> pd.DataFrame:
    user_vec, liked_titles = build_user_profile_vector(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        tfidf_matrix=tfidf_matrix,
        movieid_to_index=movieid_to_index,
        min_rating=min_rating
    )
    if user_vec is None:
        return pd.DataFrame(), []

    # similarity between user vector and all movies
    sims = cosine_similarity([user_vec], tfidf_matrix).ravel()

    out = movies.copy()
    out["similarity_score"] = sims

    if exclude_rated:
        rated_movie_ids = set(ratings.loc[ratings["userId"] == user_id, "movieId"].astype(int).tolist())
        out = out[~out["movieId"].astype(int).isin(rated_movie_ids)]

    out = out.sort_values("similarity_score", ascending=False).head(top_n).copy()
    return out, liked_titles