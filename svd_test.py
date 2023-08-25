import unittest
import numpy as np
from svd import top_q_eig

class TestQSVD(unittest.TestCase):
    """Unit tests for the top_q_eig function."""

    def test_diag_matrix(self) -> None:
        """Tests top_q_eig with a diagonal matrix."""
        matrix = np.diag(np.array([1., 2., 3., 4., 5.,
                                   6., 7., 8., 9., 10.]))
        q = 2
        eigen_system = top_q_eig(matrix, q)

        expected_vals = np.array([10., 9., 8.])
        expected_vecs = np.array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]
        ]).T

        self.assertTrue(
            np.allclose(eigen_system.values, expected_vals),
            "Eigenvalues mismatch"
        )
        self.assertTrue(
            np.allclose(eigen_system.min_value, expected_vals[-1]),
            "Min eigenvalue mismatch"
        )
        self.assertTrue(
            np.allclose(expected_vecs, eigen_system.vectors(), atol=1e-03),
            "Eigenvectors mismatch"
        )

    def test_nondiagonal_matrix(self) -> None:
        """Tests top_q_eig with a non-diagonal matrix."""
        matrix = np.array([
            [3., 1., 1., 1.],
            [1., 2., 1., 1.],
            [1., 1., 3., 1.],
            [1., 1., 1., 2.]
        ])
        q = 3
        eigen_system = top_q_eig(matrix, q)

        val_1 = 1/2 * (7 + np.sqrt(17))
        val_2 = 1/2 * (7 - np.sqrt(17))
        expected_vals = np.array([val_1, 2., val_2, 1.])

        vec_1 = [1/4 * (1 + np.sqrt(17)), 1, 1/4 * (1 + np.sqrt(17)), 1.]
        vec_2 = [-1., 0., 1., 0.]
        vec_3 = [1/4 * (1 - np.sqrt(17)), 1., 1/4 * (1 - np.sqrt(17)), 1.]
        expected_vecs = np.array([vec_1, vec_2, vec_3]).T
        expected_vecs /= np.linalg.norm(expected_vecs, axis=0)

        self.assertTrue(
            np.allclose(eigen_system.values, expected_vals),
            "Eigenvalues mismatch"
        )
        self.assertTrue(
            np.allclose(eigen_system.min_value, expected_vals[-1]),
            "Min eigenvalue mismatch"
        )
        self.assertTrue(
            np.allclose(np.abs(expected_vecs), np.abs(eigen_system.vectors()), atol=1e-03),
            "Eigenvectors mismatch"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
