import torch
from eigenpro.utils.types import assert_and_raise

class SingleDeviceTensor(torch.Tensor):
    pass

class DistributedTensor:

    def __init__(self, tensor_list):
        assert_and_raise(tensor_list, list)
        self._list = tensor_list

    def __getitem__(self, index):
        return self._list[index]

    def __len__(self):
        return len(self._list)

    def __add__(self, tensor):
        assert_and_raise(tensor, DistributedTensor)
        return DistributedTensor([self[i]+tensor[i] for i in len(self)])
        
    def matrix_multiply(self, tensor):
        assert_and_raise(tensor, DistributedTensor)
        return DistributedTensor([self[i] @ tensor[i] for i in len(self)])

    def matrix_multiply_and_add(self, mult_tensor, add_tensor, **kwargs):
        assert_and_raise(mult_tensor, DistributedTensor)
        assert_and_raise(add_tensor, DistributedTensor)
        return DistributedTensor([torch.addmm(add_tensor[i], self[i], mult_tensor[i], **kwargs) for i in len(self)])

    @property
    def T(self):
        return DistributedTensor([t.T for t in self])



if __name__ == "__main__":
    DistributedTensor([torch.zeros(10,10)])