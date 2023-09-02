import unittest
import torch
from models import KernelMachine

def linear_kernel(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1.matmul(x2.t())

class TestKernelMachine(unittest.TestCase):

    def setUp(self):
        self.kernel_fn = linear_kernel
        self.n_outputs = 2
        self.centers = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.weights = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        self.machine = KernelMachine(self.kernel_fn, self.n_outputs,
                                     self.centers, self.weights)

    def test_n_outputs(self):
        self.assertEqual(self.machine.n_outputs, self.n_outputs)

    def test_n_centers(self):
        self.assertEqual(self.machine.n_centers, 2)

    def test_forward(self):
        x = torch.tensor([[1.0, 1.0]])
        expected_output = torch.tensor([[6.5, 8.5]])
        torch.testing.assert_close(self.machine.forward(x), expected_output)

    def test_add_centers(self):
        new_centers = torch.tensor([[5.0, 6.0]])
        new_weights = torch.tensor([[0.5, 0.5]])
        self.machine.add_centers(new_centers, new_weights)
        self.assertEqual(self.machine.n_centers, 3)

    def test_shallow_copy(self):
        copy_machine = self.machine.shallow_copy()
        self.assertEqual(copy_machine.n_outputs, self.machine.n_outputs)
        self.assertEqual(copy_machine.n_centers, self.machine.n_centers)
        torch.testing.assert_close(
            copy_machine.forward(torch.tensor([[1.0, 1.0]])),
            self.machine.forward(torch.tensor([[1.0, 1.0]])))

    def test_weights(self):
        expected_weights = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        torch.testing.assert_close(self.machine.weights, expected_weights)
        
    def test_update_by_index(self):
        indices = torch.tensor([0])
        delta = torch.tensor([[0.1, 0.2]])
        self.machine.update_by_index(indices, delta)
        # The weights at index 0 should be updated
        updated_weights = torch.tensor([[1.1, 0.7], [0.5, 1.0]]) 
        torch.testing.assert_close(self.machine.weights, updated_weights)

if __name__ == "__main__":
    unittest.main()
