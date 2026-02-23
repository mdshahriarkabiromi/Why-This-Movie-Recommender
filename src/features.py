from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf(movies, max_features=5000, stop_words="english"):
    """
    Build TF-IDF matrix from combined movie text (genres + tags).
    """
    if "combined_text" not in movies.columns:
        raise ValueError("combined_text column missing. Run preprocessing first.")

    tfidf = TfidfVectorizer(
    max_features=max_features,
    stop_words=stop_words,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.60
    )

    tfidf_matrix = tfidf.fit_transform(movies["combined_text"])

    return tfidf, tfidf_matrix