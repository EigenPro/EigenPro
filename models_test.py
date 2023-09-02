import unittest
import torch
from models import ShardedKernelMachine, PreallocatedKernelMachine
from device import Device


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

        self.device = Device.create()
        self.kms = []



        for d in self.device.devices:
          centers = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5., 6.0]])
          weights = torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.5, 0.5]])
          self.kms.append( PreallocatedKernelMachine(self.kernel_fn, self.n_outputs ,centers,weights=weights,device= d) )


        self.ShardedKernelMachine = ShardedKernelMachine(self.kms)

        centers = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5., 6.0]])
        weights = torch.tensor([[1.0, 0.5], [0.5, 1.0], [0.5, 0.5]])
        self.km_sample = PreallocatedKernelMachine(self.kernel_fn, self.n_outputs, centers, weights=weights)

    def are_tensor_lists_equal(self,list1, list2):
        if len(list1) != len(list2):
            return False
        for tensor1, tensor2 in zip(list1, list2):
            if not torch.equal(tensor1, tensor2):
                return False
        return True
    ####checking PreallocatedKernelMachine
    def test_n_outputs(self):
        self.assertEqual(self.km_sample.n_outputs, self.n_outputs)

    def test_n_centers(self):
        self.assertEqual(self.km_sample.n_centers, 3)

    def test_forward(self):
        x = torch.tensor([[1.0, 1.0]])
        expected_output = torch.tensor([[12., 14.]])
        torch.testing.assert_close(self.km_sample.forward(x), expected_output)

    def test_add_centers(self):
        new_centers = torch.tensor([[5.0, 6.0]])
        new_weights = torch.tensor([[0.5, 0.5]])
        self.km_sample.add_centers(new_centers, new_weights)
        self.assertEqual(self.km_sample.n_centers, 4)

    ####checking ShardedKernelMachine
    def test_forward_sharded(self):
        x = torch.tensor([[1.0, 1.0]])
        out = self.ShardedKernelMachine.forward(x)
        expected_output = []
        for d in self.device.devices:
          expected_output.append(torch.tensor([[12, 14]]).to(d))
        self.are_tensor_lists_equal(out, expected_output)
    def test_add_centers_sharded(self):
        new_centers = torch.tensor([[7.0, 8.0]])
        new_centers_list = self.device(new_centers,strategy = "divide_to_gpu")
        n_devices = len(self.device.devices)
        self.ShardedKernelMachine.add_centers(new_centers_list)
        for i,_ in enumerate(self.device.devices):
          self.assertEqual(self.ShardedKernelMachine.shard_kms[i].n_centers, 3+(i+1)//n_devices)

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
    unittest.main(verbosity=2)