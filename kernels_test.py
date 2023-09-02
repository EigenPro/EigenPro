import unittest
import torch
from kernels import euclidean

class TestEuclidean(unittest.TestCase):

    def test_basic_case(self):
        samples = torch.tensor([[1, 2], [3, 4]]).float()
        centers = torch.tensor([[1, 2], [2, 3]]).float()
        expected_output_sq = torch.tensor([[0., 2.], [8., 2.]])
        expected_output_no_sq = torch.tensor(
            [[0., torch.sqrt(torch.tensor(2.))],
             [torch.sqrt(torch.tensor(8.)), torch.sqrt(torch.tensor(2.))]])
        result_sq = euclidean(samples, centers)
        result_no_sq = euclidean(samples, centers, squared=False)
        self.assertTrue(torch.allclose(result_sq, expected_output_sq))
        self.assertTrue(torch.allclose(result_no_sq, expected_output_no_sq))

    def test_1D_tensors(self):
        samples = torch.tensor([[1, 2, 3]]).float()
        centers = torch.tensor([[1, 2, 3]]).float()
        expected_output_sq = torch.tensor([0.])
        expected_output_no_sq = torch.tensor([0.])
        result_sq = euclidean(samples, centers)
        result_no_sq = euclidean(samples, centers, squared=False)
        self.assertTrue(torch.allclose(result_sq, expected_output_sq))
        self.assertTrue(torch.allclose(result_no_sq, expected_output_no_sq))

    def test_different_size(self):
        samples = torch.tensor([[1, 2], [3, 4],[5,6]]).float()
        centers = torch.tensor([[1, 2], [3, 4]]).float()
        expected_output_sq = torch.tensor([[0.,8.],[8.,0.],[32.,8]])
        expected_output_no_sq = torch.sqrt( expected_output_sq )
        result_sq = euclidean(samples, centers)
        result_no_sq = euclidean(samples, centers, squared=False)
        self.assertTrue(torch.allclose(result_sq, expected_output_sq))
        self.assertTrue(torch.allclose(result_no_sq, expected_output_no_sq))

    def test_same_samples_and_centers(self):
        samples = torch.tensor([[1, 2], [3, 4]]).float()
        centers = samples
        expected_output_sq = torch.tensor([[0.,8.],[8.,0.]])
        expected_output_no_sq = torch.sqrt( expected_output_sq )
        result_sq = euclidean(samples, centers)
        result_no_sq = euclidean(samples, centers, squared=False)
        self.assertTrue(torch.allclose(result_sq, expected_output_sq))
        self.assertTrue(torch.allclose(result_no_sq, expected_output_no_sq))


if __name__ == '__main__':
    unittest.main(verbosity=2)



