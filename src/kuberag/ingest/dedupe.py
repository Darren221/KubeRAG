import math

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def nearest_similarity(
    embedding: list[float],
    existing_embeddings: list[list[float]],
) -> tuple[float, int | None]:
    if not existing_embeddings:
        return -math.inf, None

    query = np.asarray(embedding, dtype=np.float64)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return 0.0, None

    matrix = np.asarray(existing_embeddings, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)

    sims = np.full(len(existing_embeddings), -np.inf, dtype=np.float64)
    valid = norms > 0
    if not valid.any():
        return 0.0, None

    sims[valid] = (matrix[valid] @ query) / (norms[valid] * query_norm)
    best_index = int(np.argmax(sims))
    return float(sims[best_index]), best_index


def is_duplicate(
    embedding: list[float],
    existing_embeddings: list[list[float]],
    threshold: float = 0.95,
) -> bool:
    similarity, _ = nearest_similarity(embedding, existing_embeddings)
    return similarity >= threshold
