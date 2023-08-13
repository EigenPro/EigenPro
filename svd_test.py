import unittest
import numpy as np
from svd import top_q_eig


class TestQSVD(unittest.TestCase):

    def test_diag_matrix(self) -> None:
        matrix = np.diag(np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]))
        q = 2
        eigenvals, eigenvecs, eigenvalq = top_q_eig(matrix, q)

        expected_output_vals = np.array([10., 9.])
        expected_output_vecs = np.array([
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]
        ]).T
        expected_eigenvalq = 8.

        self.assertTrue(np.allclose(eigenvals, expected_output_vals))
        self.assertTrue(np.allclose(eigenvalq, expected_eigenvalq))
        self.assertTrue(np.allclose(expected_output_vecs, eigenvecs, atol=1e-03))

    def test_nondiagonal_matrix(self) -> None:
        matrix = np.array([
            [3., 1., 1., 1.],
            [1., 2., 1., 1.],
            [1., 1., 3., 1.],
            [1., 1., 1., 2.]
        ])
        q = 3
        eigenvals, eigenvecs, eigenvalq = top_q_eig(matrix, q)

        expected_output_vals = np.array([
            1/2 * (7 + np.sqrt(17)),
            2.,
            1/2 * (7 - np.sqrt(17))
        ])
        expected_output_vecs = np.array([
            [1/4 * (1 + np.sqrt(17)), 1, 1/4 * (1 + np.sqrt(17)), 1.],
            [-1., 0., 1., 0.],
            [1/4 * (1 - np.sqrt(17)), 1., 1/4 * (1 - np.sqrt(17)), 1.]
        ]).T

        expected_output_vecs /= np.linalg.norm(expected_output_vecs, axis=0)
        expected_eigenvalq = 1.

        self.assertTrue(np.allclose(eigenvals, expected_output_vals))
        self.assertTrue(np.allclose(eigenvalq, expected_eigenvalq))
        self.assertTrue(np.allclose(np.abs(expected_output_vecs), np.abs(eigenvecs), atol=1e-03))


if __name__ == '__main__':
    unittest.main(verbosity=2)
