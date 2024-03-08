import torch
from eigenpro.utils.types import assert_and_raise
from typing import List

class SingleDeviceTensor(torch.Tensor):
    def at_device(self, device_idx):
        return self

    def __repr__(self):
        return f'On device {self.device}\n  ' + super().__repr__()


class BaseDeviceTensor(SingleDeviceTensor):
    pass


class DistributedTensor:

    def __init__(self, tensor_list: List[SingleDeviceTensor], base_device_idx: int):
        self._list = [BaseDeviceTensor(t) if i==base_device_idx else SingleDeviceTensor(t) for i,t in enumerate(tensor_list)]
        self._base_device_idx = base_device_idx
        
        self._lengths = torch.as_tensor([len(tensor) for tensor in tensor_list]).to(self.parts[base_device_idx])
        self._total_length = sum(self._lengths)
        self._offsets = torch.cumsum(torch.as_tensor(self._lengths), 0, dtype=torch.int64) - self._lengths[0]

    def __getitem__(self, index):
        # broadcasting assumes all indices are on the same device
        device_id_of_index = self.get_device_id_by_idx(index)
        return SingleDeviceTensor(self._list[device_id_of_index][index - self._offsets[device_id_of_index]])

    def get_device_id_by_idx(self, index):
        return torch.searchsorted(self._offsets, index if isinstance(index, int) else index[0]) - 1

    @property
    def lengths(self):
        return self._lengths

    @property
    def total_length(self):
        return self._total_length

    @property
    def num_parts(self):
        return len(self._list)

    @property
    def base_device_idx(self):
        return self._base_device_idx

    @property
    def parts(self):
        return self._list

    def __len__(self):
        return self._total_length

    def __add__(self, tensor):
        assert_and_raise(tensor, DistributedTensor)
        return DistributedTensor([part + tensor.parts[i] for i, part in enumerate(self.parts)])

    def __sub__(self, tensor):
        assert_and_raise(tensor, DistributedTensor)
        return DistributedTensor([part - tensor.parts[i] for i, part in enumerate(self.parts)])

    def __matmul__(self, tensor):
        assert_and_raise(tensor, DistributedTensor)
        return DistributedTensor([part @ tensor.parts[i] for i, part in enumerate(self.parts)])

    def __mul__(self, scalar):
        assert_and_raise(scalar, float)
        return DistributedTensor([part * scalar for i, part in enumerate(self.parts)])

    def __pow__(self, exponent):
        assert_and_raise(exponent, int)
        return DistributedTensor([part ** exponent for i, part in enumerate(self.parts)])

    def __repr__(self):
        string = f"{type(self).__name__}\n"
        for i, p in enumerate(self.parts): 
            string += (
                f'* {i}: ' + str(p) + '\n' if i==self.base_device_idx
            else f'  {i}: ' + str(p) + '\n')
        return string

    @property
    def shape(self):
        return [part.shape for part in self.parts]

    @property
    def T(self):
        return DistributedTensor([part.T for part in self.parts])

class BroadcastTensor(DistributedTensor):
    def at_device(self, device_idx):
        return self.parts[device_idx]

    def __repr__(self):
        return "BroadcastTensor\n" + str(self.parts[0]) 

class ScatteredTensor(DistributedTensor):
    # def at_device(self, device_idx):
    #     return self.parts[device_idx]
    pass


if __name__ == "__main__":
    from eigenpro.device_manager import DeviceManager
    device_manager = DeviceManager([torch.device('cpu'), torch.device('cpu')], base_device_idx=0)
    tensor = DistributedTensor(torch.arange(40).reshape(20,2).split(10), base_device_idx=1)
    
    # index = DistributedTensor([torch.LongTensor([7,8]), torch.LongTensor(112,13)])
    print(tensor)
    index1 = SingleDeviceTensor(torch.LongTensor([7,8]))
    index2 = SingleDeviceTensor(torch.LongTensor([12,13]))
    index3 = BaseDeviceTensor(torch.LongTensor([11,9]))
    print(tensor[index1])
    print(tensor[index2])
    print(tensor[index3])