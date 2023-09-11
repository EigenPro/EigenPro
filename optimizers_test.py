import numpy as np
import unittest
import torch

import models
import optimizers as opt
import preconditioner as pcd
import svd


def linear_kernel_fn(a, b):
    return torch.matmul(a, b.T)


class TestTensorFunctions(unittest.TestCase):

    def test_split_ids(self):
        # Test with positive, negative, and zero values
        ids = torch.tensor([1, 2, 3, -1, -2, -3, 0])
        split_id = 0
        leq_ids, g_ids = opt.split_ids(ids, split_id)
        self.assertTrue(torch.equal(leq_ids, torch.tensor([-1, -2, -3, 0])))
        self.assertTrue(torch.equal(g_ids, torch.tensor([1, 2, 3])))

        # Test with all ids less than split_id
        ids = torch.tensor([-1, -2, -3])
        leq_ids, g_ids = opt.split_ids(ids, split_id)
        self.assertTrue(torch.equal(leq_ids, torch.tensor([-1, -2, -3])))
        self.assertTrue(torch.equal(g_ids, torch.tensor([])))

        # Test with all ids greater than split_id
        ids = torch.tensor([1, 2, 3])
        leq_ids, g_ids = opt.split_ids(ids, split_id)
        self.assertTrue(torch.equal(leq_ids, torch.tensor([])))
        self.assertTrue(torch.equal(g_ids, torch.tensor([1, 2, 3])))

        # Test with empty tensor
        ids = torch.tensor([])
        leq_ids, g_ids = opt.split_ids(ids, split_id)
        self.assertTrue(torch.equal(leq_ids, torch.tensor([])))
        self.assertTrue(torch.equal(g_ids, torch.tensor([])))

    def test_obtain_by_ids(self):
        # Test with multiple tensors
        t1 = torch.tensor([1, 2, 3])
        t2 = torch.tensor([4, 5, 6])
        t3 = torch.tensor([7, 8, 9])
        batch_ids = torch.tensor([0, 2])

        res1, res2, res3 = opt.obtain_by_ids(batch_ids, t1, t2, t3)
        self.assertTrue(torch.equal(res1, torch.tensor([1, 3])))
        self.assertTrue(torch.equal(res2, torch.tensor([4, 6])))
        self.assertTrue(torch.equal(res3, torch.tensor([7, 9])))

        # Test with single tensor
        t1 = torch.tensor([1, 2, 3])
        batch_ids = torch.tensor([0, 2])
        res1 = opt.obtain_by_ids(batch_ids, t1)
        self.assertTrue(torch.equal(res1, torch.tensor([1, 3])))

        # Test with empty batch_ids
        t1 = torch.tensor([1, 2, 3])
        batch_ids = torch.tensor([])
        res1 = opt.obtain_by_ids(batch_ids, t1)
        self.assertTrue(torch.equal(res1, torch.tensor([])))

        # Test with empty tensor
        t1 = torch.tensor([])
        batch_ids = torch.tensor([0, 2])
        res1 = opt.obtain_by_ids(batch_ids, t1)
        self.assertTrue(torch.equal(res1, torch.tensor([])))

        # Test with no tensors provided
        batch_ids = torch.tensor([0, 2])
        with self.assertRaises(ValueError):
            opt.obtain_by_ids(batch_ids)

        # Test with batch_ids same length as tensor
        t1 = torch.tensor([1, 2, 3])
        batch_ids = torch.tensor([0, 1, 2])
        res1 = opt.obtain_by_ids(batch_ids, t1)
        self.assertTrue(torch.equal(res1, torch.tensor([1, 2, 3])))

    def setUp(self):
        self.kernel_fn = linear_kernel_fn
        self.n_outputs = 2
        self.centers = torch.tensor([[1.0, 2.0],
                                     [3.0, 4.0]], dtype=torch.float32)
        self.weights = torch.tensor([[0.0, 0.0],
                                     [0.0, 0.0]], dtype=torch.float32)
        self.model = models.BlockKernelMachine(self.kernel_fn, self.n_outputs,
                                               self.centers, self.weights)
        self.threshold_index = 1

        # Creating a mock EigenSystem
        self.eigenvalues = np.asarray([3.0, 2.0, 1.0], dtype=np.float32)
        self.eigenvectors = np.asarray([[1.0, 0.0],
                                       [0.0, 1.0]], dtype=np.float32)
        self.eigensys = svd.EigenSystem(self.eigenvalues, self.eigenvectors)
        self.eigensys = pcd.KernelEigenSystem(self.eigensys, 1.0)
        
        self.top_q_eig = 1
        self.pweights = torch.zeros([self.centers.shape[0],
                                     self.model.n_outputs])
        self.precon = pcd.Preconditioner(self.model._kernel_fn, self.centers,
                                         self.pweights, self.top_q_eig)
        self.optimizer = opt.EigenPro(self.model, self.threshold_index,
                                      self.precon)

    def test_init(self):
        self.assertIsNotNone(self.optimizer.model)
        self.assertIsNotNone(self.optimizer.precon)
    
    def test_step_with_in_batch(self):
        batch_x = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
        batch_y = torch.tensor([[2.0, 3.0], [3.0, 4.0]], dtype=torch.float32)
        batch_ids = torch.tensor([0, 1], dtype=torch.int64)

        np.testing.assert_array_almost_equal(self.model.weights,
                                             np.array([[0.0, 0.0],
                                                       [0.0, 0.0]]), decimal=6)
        batch_p = self.model(batch_x)
        self.optimizer.step(batch_x, batch_y, batch_ids)        
        np.testing.assert_array_almost_equal(self.model.weights,
                                             np.array([[0.08, 0.12],
                                                       [0.12, 0.16]]),
                                             decimal=6)

# Running the test
if __name__ == '__main__':
    unittest.main()


# Run the unit tests
unittest.main(argv=[''], verbosity=2, exit=False)