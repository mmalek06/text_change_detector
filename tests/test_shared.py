import numpy as np

from text_change_detector.shared.graph import knn_sparsify


class TestKnnSparsify:
    def test_keeps_top_k_and_symmetrises(self):
        matrix = np.array([
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.3, 0.2],
            [0.2, 0.3, 1.0, 0.8],
            [0.1, 0.2, 0.8, 1.0],
        ])
        adjacency = knn_sparsify(matrix, k=1)

        assert np.allclose(adjacency, [
            [0.0, 0.9, 0.0, 0.0],
            [0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.8],
            [0.0, 0.0, 0.8, 0.0],
        ])

    def test_edge_survives_when_only_one_side_selects_it(self):
        matrix = np.array([
            [1.0, 0.5, 0.4],
            [0.5, 1.0, 0.9],
            [0.4, 0.9, 1.0],
        ])
        adjacency = knn_sparsify(matrix, k=1)

        assert np.allclose(adjacency, [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.9],
            [0.0, 0.9, 0.0],
        ])

    def test_diagonal_is_zeroed_and_result_symmetric(self):
        matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        adjacency = knn_sparsify(matrix, k=1)

        assert np.allclose(np.diag(adjacency), [0.0, 0.0])
        assert np.allclose(adjacency, adjacency.T)
