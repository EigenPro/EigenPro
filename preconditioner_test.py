"""Tests for the module `preconditioner.py`.
"""

import unittest
import numpy as np
import torch
from utils import DEFAULT_DTYPE

import preconditioner as pcd

torch.set_default_dtype(DEFAULT_DTYPE)

def linear_kernel(centers: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
    """ Dummy kernel for testing.
    Attributes:
        centers: tensor of shape (n, d)
        samples: tensor of shape (m, d)
    """
    return centers.matmul(samples.t())


class MockEigenSystem:
    def __init__(self, values, vectors, min_value):
        self.values = values
        self.vectors = vectors
        self.min_value = min_value


class TestPreconditionerAndKernelEigenSystem(unittest.TestCase):

    def test_KernelEigenSystem(self):
        # Initialize a mock EigenSystem
        mock_eigensys = MockEigenSystem(
            values=np.array([3, 2]),
            vectors=np.array([[1, 0], [0, 1]]), min_value=.1)

        # Initialize a KernelEigenSystem based on the mock EigenSystem
        kernel_eigensys = pcd.KernelEigenSystem(mock_eigensys, beta=10)

        # Test beta property
        self.assertEqual(kernel_eigensys.beta, 10)

        # Test normalized_ratios property
        np.testing.assert_array_almost_equal(
            kernel_eigensys.normalized_ratios,
            np.array([0.322222, 0.475]), decimal=6)

    def test_learning_rate(self):
        centers = torch.Tensor([[1, 2, 3], [3, 4, 1], [-1, 0.5, 2.1]])
        weights = torch.Tensor([[1.0], [2.0], [3.0]])
        top_q_eig = 2
        precon = pcd.Preconditioner(linear_kernel, centers, weights, top_q_eig)

        self.assertEqual(precon.critical_batch_size, 276)
        self.assertAlmostEqual(precon.learning_rate(1), 0.01923, places=4)
        self.assertAlmostEqual(precon.learning_rate(10), 0.19230, places=4)
        self.assertAlmostEqual(precon.learning_rate(100), 1.92307, places=4)
        self.assertAlmostEqual(precon.learning_rate(276), 5.32412, places=4)
        self.assertAlmostEqual(precon.learning_rate(10000), 10.35707, places=4)
        self.assertAlmostEqual(precon.learning_rate(100000), 10.61337, places=4)

    def test_delta_and_update(self):
        centers = torch.Tensor([[1, 2, 3, 2],
                                [3, 4, 1, 1],
                                [-1, 0.5, 2.1, 0]])
        weights = torch.Tensor([[1.0],
                                [2.0],
                                [0.0]])
        top_q_eig = 1
        precon = pcd.Preconditioner(linear_kernel, centers, weights, top_q_eig)

        batch_x = torch.Tensor([[1, 2, 0, 0.2],
                                [3, 4, -1, 0.6]])
        grad = torch.Tensor([[0.5],
                             [0.6]])

        delta,_ = precon.delta(batch_x, grad)
        np.testing.assert_array_almost_equal(
            delta, np.array([[0.718163],
                             [0.913501],
                             [0.162083]]), decimal=6)

        # Verify updated weights
        precon.update(delta, 2)
        np.testing.assert_array_almost_equal(
            precon._weights, np.array([[1.026599],
                                       [2.033834],
                                       [0.006003]]), decimal=6)


unittest.main(argv=[''], verbosity=2, exit=False)
