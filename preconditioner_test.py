import numpy as np
import torch
import unittest

import preconditioner as pcd



def linear_kernel(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1.matmul(x2.t())


class MockEigenSystem:
    def __init__(self, values, vectors, min_value):
        self._values = values
        self._vectors = vectors
        self._min_value = min_value


class TestPreconditionerAndKernelEigenSystem(unittest.TestCase):
    
    def test_KernelEigenSystem(self):
        # Initialize a mock EigenSystem
        mock_eigensys = MockEigenSystem(
            values=np.array([3, 2, 0.1]),
            vectors=np.array([[1, 0], [0, 1]]), min_value=1)
        
        # Initialize a KernelEigenSystem based on the mock EigenSystem
        kernel_eigensys = pcd.KernelEigenSystem(mock_eigensys, beta=10, 
                                                scale=0.1)
        
        # Test beta property
        self.assertEqual(kernel_eigensys.beta, 10)
        
        # Test normalized_ratios property
        np.testing.assert_array_almost_equal(
            kernel_eigensys.normalized_ratios,
            np.array([0.032222, 0.0475]), decimal=6)

    def test_learning_rate(self):
        centers = torch.tensor([[1, 2, 3], [3, 4, 1], [-1, 0.5, 2.1]],
                               dtype=torch.float32)
        weights = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        top_q_eig = 2
        precon = pcd.Preconditioner(linear_kernel, centers, weights, top_q_eig)
        
        self.assertEqual(precon.critical_batch_size, 92)
        self.assertAlmostEqual(precon.learning_rate(1), 0.01923, places=4)
        self.assertAlmostEqual(precon.learning_rate(10), 0.19230, places=4)
        self.assertAlmostEqual(precon.learning_rate(92), 1.78114, places=4)
        self.assertAlmostEqual(precon.learning_rate(100), 1.85503, places=4)
        self.assertAlmostEqual(precon.learning_rate(1000), 3.25088, places=4)
        self.assertAlmostEqual(precon.learning_rate(10000), 3.51539, places=4)
        
    def test_delta_and_update(self):
        centers = torch.tensor([[1, 2, 3, 2],
                                [3, 4, 1, 1],
                                [-1, 0.5, 2.1, 0]],
                               dtype=torch.float32)
        weights = torch.tensor([[1.0],
                                [2.0],
                                [0.0]], dtype=torch.float32)
        top_q_eig = 1
        precon = pcd.Preconditioner(linear_kernel, centers, weights, top_q_eig)
        
        batch_x = torch.tensor([[1, 2, 0, 0.2],
                                [3, 4, -1, 0.6]], dtype=torch.float32)
        grad = torch.tensor([[0.5],
                             [0.6]], dtype=torch.float32)
        
        delta = precon.delta(batch_x, grad)
        np.testing.assert_array_almost_equal(
            delta, np.array([[0.079796],
                             [0.1015  ],
                             [0.018009]]), decimal=6)
        
        # Verify updated weights
        precon.update(delta, 2)
        np.testing.assert_array_almost_equal(
            precon._weights, np.array([[1.004339e+00],
                                       [2.005519e+00],
                                       [9.792851e-04]]), decimal=6)

unittest.main(argv=[''], verbosity=2, exit=False)