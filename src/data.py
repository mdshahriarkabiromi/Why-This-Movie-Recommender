import pandas as pd


def load_data(movies_path, ratings_path, tags_path=None):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    if tags_path:
        tags = pd.read_csv(tags_path)
        movies = merge_tags(movies, tags)

    movies = preprocess_movies(movies)

    return movies, ratings


def merge_tags(movies, tags):
    """
    Aggregate tags per movie and merge into movies dataframe.
    """
    tags_grouped = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )

    movies = movies.merge(tags_grouped, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")

    return movies

def preprocess_movies(movies):
    """
    Clean genres and create combined text field.
    """
    movies["genres"] = movies["genres"].fillna("")
    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

    if "tag" not in movies.columns:
        movies["tag"] = ""
    movies["combined_text"] = movies["genres"] + " " + movies["tag"]

    return movies