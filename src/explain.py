# src/explain.py
# -----------------------------
# Explanation utilities for "Why This Movie?"
# Produces natural-language reasons based on overlapping TF-IDF tokens.
# Designed to:
#   - work with sparse TF-IDF matrices
#   - deduplicate keywords
#   - avoid awkward "genre-genre" bigrams like "animation children"
#   - prefer meaningful unigrams/tags (e.g., "pixar") and only keep bigrams if useful
# -----------------------------

import re
import numpy as np


# Words that are often too generic (especially when forming awkward bigrams)
GENERIC = {
    "film", "movie", "movies",
    "adventure", "comedy", "drama", "action", "thriller",
    "romance", "horror", "fantasy", "sci", "fi", "scifi",
    "children", "animation"
}


def _clean_token(token: str) -> str:
    """
    Normalize spacing/case and collapse duplicated bigrams like 'fantasy fantasy' -> 'fantasy'.
    """
    token = token.lower().strip()
    token = re.sub(r"\s+", " ", token)

    parts = token.split()
    if len(parts) == 2 and parts[0] == parts[1]:
        return parts[0]

    return token


def _is_good_bigram(token: str) -> bool:
    """
    Keep bigrams only if they look meaningful.
    Reject bigrams made mostly of generic genre words like 'animation children'.
    """
    parts = token.split()
    if len(parts) != 2:
        return True

    a, b = parts

    # Reject if both words are generic genre-ish tokens
    if a in GENERIC and b in GENERIC:
        return False

    # Reject if either token is too short (noise)
    if len(a) <= 2 or len(b) <= 2:
        return False

    return True


def _top_overlap_keywords(tfidf, base_vec, rec_vec, top_k: int = 6):
    """
    Extract top overlapping TF-IDF tokens between base and recommended movies.
    Uses elementwise product of their vectors as an overlap score.
    """
    feature_names = tfidf.get_feature_names_out()

    # Compute overlap importance (sparse-safe)
    if hasattr(base_vec, "multiply"):  # scipy sparse
        overlap = base_vec.multiply(rec_vec).toarray().ravel()
    else:
        overlap = np.multiply(np.asarray(base_vec).ravel(), np.asarray(rec_vec).ravel())

    top_idx = np.argsort(overlap)[::-1]

    keywords = []
    seen = set()

    for i in top_idx:
        if overlap[i] <= 0:
            break

        token = _clean_token(feature_names[i])

        if not token or token in seen:
            continue

        # filter awkward bigrams
        if not _is_good_bigram(token):
            continue

        # avoid near-duplicate phrasing:
        # if 'pixar' already used, skip 'pixar animation' etc.
        parts = token.split()
        if len(parts) == 2 and (parts[0] in seen or parts[1] in seen):
            continue

        # skip ultra generic tokens entirely if they appear alone
        if len(parts) == 1 and token in {"film", "movie", "movies"}:
            continue

        seen.add(token)
        keywords.append(token)

        if len(keywords) >= top_k:
            break

    # Prefer unigrams in final output if available
    unigrams = [k for k in keywords if len(k.split()) == 1]
    bigrams = [k for k in keywords if len(k.split()) == 2]

    final = (unigrams[:top_k] + bigrams[:max(0, top_k - len(unigrams))])[:top_k]
    return final


def _naturalize_keywords(keywords):
    """
    Turn keywords into a clean sentence.
    """
    if not keywords:
        return "It matches your preferences based on overall content similarity."

    if len(keywords) == 1:
        return f"It strongly matches on **{keywords[0]}**."
    if len(keywords) == 2:
        return f"It matches on **{keywords[0]}** and **{keywords[1]}**."
    return f"It matches on themes like **{keywords[0]}**, **{keywords[1]}**, and **{keywords[2]}**."


def explain_movie_to_movie(
    base_title: str,
    rec_title: str,
    movies,
    tfidf,
    tfidf_matrix,
    title_index,
    top_k: int = 6
) -> str:
    """
    Explain recommendation when user selected a single base movie.
    """
    base_idx = int(title_index[base_title])
    rec_idx = int(title_index[rec_title])

    base_vec = tfidf_matrix[base_idx]
    rec_vec = tfidf_matrix[rec_idx]

    keywords = _top_overlap_keywords(tfidf, base_vec, rec_vec, top_k=top_k)
    reason = _naturalize_keywords(keywords)

    return (
        f"Because you selected **{base_title}**, this recommendation is similar in content. "
        f"{reason}"
    )


def explain_user_to_movie(
    rec_title: str,
    movies,
    tfidf,
    tfidf_matrix,
    title_index,
    liked_titles,
    top_k: int = 6
) -> str:
    """
    Explain recommendation using user's top liked movies.
    Picks the most similar liked movie and extracts overlapping keywords.
    """
    rec_idx = int(title_index[rec_title])
    rec_vec = tfidf_matrix[rec_idx]

    if not liked_titles:
        return f"Recommended based on your overall viewing preferences. {_naturalize_keywords([])}"

    candidates = liked_titles[:5]
    best_title = None
    best_score = -1.0

    for t in candidates:
        if t not in title_index:
            continue

        b_idx = int(title_index[t])
        b_vec = tfidf_matrix[b_idx]

        # cosine similarity between two sparse rows (manual, sparse-safe)
        if hasattr(b_vec, "multiply"):
            num = float(b_vec.multiply(rec_vec).sum())
            den = float(np.sqrt(b_vec.multiply(b_vec).sum()) * np.sqrt(rec_vec.multiply(rec_vec).sum()))
            score = (num / den) if den != 0 else 0.0
        else:
            b = np.asarray(b_vec).ravel()
            r = np.asarray(rec_vec).ravel()
            score = float((b @ r) / (np.linalg.norm(b) * np.linalg.norm(r) + 1e-9))

        if score > best_score:
            best_score = score
            best_title = t

    if best_title and best_title in title_index:
        b_idx = int(title_index[best_title])
        b_vec = tfidf_matrix[b_idx]

        keywords = _top_overlap_keywords(tfidf, b_vec, rec_vec, top_k=top_k)
        reason = _naturalize_keywords(keywords)

        return f"Recommended because your profile resembles someone who liked **{best_title}**. {reason}"

    return f"Recommended based on your overall viewing preferences. {_naturalize_keywords([])}"