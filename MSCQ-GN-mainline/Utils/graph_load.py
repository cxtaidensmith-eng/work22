import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def feature_to_adj(feature, k_nearest_neighobrs=7, rm_common_neighbors=2):
    """Build the fixed kNN prior used by the learnable subject graph."""
    neighbors = NearestNeighbors(
        n_neighbors=k_nearest_neighobrs + 1,
        algorithm='ball_tree',
    ).fit(feature)
    adjacency = construct_symmetric_matrix(neighbors.kneighbors_graph(feature).toarray())
    adjacency = sp.csc_matrix(adjacency)
    adjacency = adjacency - sp.dia_matrix(
        (adjacency.diagonal()[np.newaxis, :], [0]),
        shape=adjacency.shape,
    )
    adjacency.eliminate_zeros()
    dense = adjacency.toarray()

    rows, cols = np.nonzero(dense)
    neighborhoods = {}
    for row, col in zip(rows, cols):
        neighborhoods.setdefault(row, []).append(col)
    for row, col in zip(rows, cols):
        if len(set(neighborhoods[row]) & set(neighborhoods[col])) < rm_common_neighbors:
            dense[row, col] = 0

    adjacency = sp.csc_matrix(dense)
    adjacency.eliminate_zeros()
    return adjacency.toarray() + sp.eye(adjacency.shape[0])


def construct_symmetric_matrix(original_matrix):
    result = np.zeros(original_matrix.shape, dtype=float)
    rows, cols = np.nonzero(original_matrix)
    result[rows, cols] = 1
    result[cols, rows] = 1
    return result
