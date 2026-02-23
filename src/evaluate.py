import numpy as np

def precision_at_k(recommended_ids, relevant_ids, k=5):
    recommended_ids = recommended_ids[:k]
    hits = len(set(recommended_ids) & set(relevant_ids))
    return hits / k