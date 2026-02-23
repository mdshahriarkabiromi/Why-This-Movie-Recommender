import streamlit as st
import joblib
import pandas as pd

from src.data import load_data, preprocess_movies
from src.recommender import (
    build_title_index,
    build_movieid_to_index,
    recommend_similar_movie,
    recommend_for_user
)
from src.explain import explain_movie_to_movie, explain_user_to_movie


st.set_page_config(page_title="Why This Movie?", page_icon="🎬", layout="centered")
st.title("🎬 Why This Movie?")

# Load raw data (same preprocessing as training)
movies, ratings = load_data(
    movies_path="data/raw/movies.csv",
    ratings_path="data/raw/ratings.csv",
    tags_path="data/raw/tags.csv"
)
movies = preprocess_movies(movies)

# Load artifacts
tfidf = joblib.load("models/artifacts/tfidf.joblib")
tfidf_matrix = joblib.load("models/artifacts/tfidf_matrix.joblib")
similarity = joblib.load("models/artifacts/similarity.joblib")

title_index = build_title_index(movies)
movieid_to_index = build_movieid_to_index(movies)

mode = st.radio(
    "Choose recommendation mode:",
    ["Similar to a movie", "Personalized (by userId)"],
    horizontal=True
)

TOP_N = st.slider("Number of recommendations", 5, 20, 10)

if mode == "Similar to a movie":
    selected = st.selectbox("Select a movie you like:", movies["title"].values)

    if st.button("Recommend"):
        recs = recommend_similar_movie(
            title=selected,
            movies=movies,
            similarity_matrix=similarity,
            title_index=title_index,
            top_n=TOP_N
        )

        for _, row in recs.iterrows():
            st.subheader(row["title"])
            st.caption(f"Similarity score: {row['similarity_score']:.3f}")
            explanation = explain_movie_to_movie(
                base_title=selected,
                rec_title=row["title"],
                movies=movies,
                tfidf=tfidf,
                tfidf_matrix=tfidf_matrix,
                title_index=title_index,
                top_k=6
            )
            st.write(explanation)
            st.divider()

else:
    # Personalized mode
    # show available user ids quickly
    user_min = int(ratings["userId"].min())
    user_max = int(ratings["userId"].max())
    user_id = st.number_input("Enter userId", min_value=user_min, max_value=user_max, value=user_min, step=1)

    min_rating = st.slider("Consider movies liked if rating ≥", 3.0, 5.0, 4.0, 0.5)

    if st.button("Recommend"):
        recs, liked_titles = recommend_for_user(
            user_id=int(user_id),
            ratings=ratings,
            movies=movies,
            tfidf_matrix=tfidf_matrix,
            movieid_to_index=movieid_to_index,
            top_n=TOP_N,
            min_rating=float(min_rating),
            exclude_rated=True
        )

        if recs.empty:
            st.error("No recommendations found for this userId (or no matching movies). Try another userId.")
        else:
            st.write("**Top liked movies used for your profile:**")
            st.write(", ".join(liked_titles) if liked_titles else "N/A")
            st.divider()

            for _, row in recs.iterrows():
                st.subheader(row["title"])
                st.caption(f"Profile similarity: {row['similarity_score']:.3f}")
                explanation = explain_user_to_movie(
                    rec_title=row["title"],
                    movies=movies,
                    tfidf=tfidf,
                    tfidf_matrix=tfidf_matrix,
                    title_index=title_index,
                    liked_titles=liked_titles,
                    top_k=6
                )
                st.write(explanation)
                st.divider()