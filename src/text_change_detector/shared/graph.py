import numpy as np


def knn_sparsify(matrix: np.ndarray, k: int = 5) -> np.ndarray:
    """Turn a dense similarity matrix into a sparse kNN adjacency matrix.

    Every node keeps an edge only to its `k` most similar neighbours; an edge
    survives if it appears in *either* endpoint's top-k (union), and the diagonal
    is zeroed. Shared by tiling and detection so both build the relation graph the
    same way.
    """
    sim = matrix.copy()

    np.fill_diagonal(sim, -np.inf)

    neighbors = np.argsort(-sim, axis=1)[:, :k]
    mask = np.zeros_like(matrix, dtype=bool)
    rows = np.arange(matrix.shape[0])[:, None]
    mask[rows, neighbors] = True
    mask |= mask.T
    adjacency = np.where(mask, matrix, 0.0)

    np.fill_diagonal(adjacency, 0.0)

    return adjacency
