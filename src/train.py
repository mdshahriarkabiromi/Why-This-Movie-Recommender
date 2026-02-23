import yaml
from sklearn.metrics.pairwise import cosine_similarity

from src.data import load_data, preprocess_movies
from src.features import build_tfidf
from src.recommender import build_title_index
from src.utils import save_model


def main(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    movies, ratings = load_data(
        cfg["data"]["movies_path"],
        cfg["data"]["ratings_path"],
        cfg["data"]["tags_path"]
    )

    movies = preprocess_movies(movies)

    tfidf, tfidf_matrix = build_tfidf(
        movies,
        max_features=cfg["model"]["max_features"],
        stop_words=cfg["model"]["stop_words"]
    )

    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    title_index = build_title_index(movies)

    save_model(tfidf, "models/artifacts/tfidf.joblib")
    save_model(tfidf_matrix, "models/artifacts/tfidf_matrix.joblib")
    save_model(similarity_matrix, "models/artifacts/similarity.joblib")
    save_model(title_index, "models/artifacts/indices.joblib")

    print("✅ Model artifacts saved.")


if __name__ == "__main__":
    main()