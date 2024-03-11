import torch
from eigenpro.utils.types import assert_and_raise
from typing import List, Union

class SingleDeviceTensor(torch.Tensor):
    def at_device(self, device_idx):
        return self

    def __repr__(self):
        return f'\nOn device {self.device}\n  ' + super().__repr__()


class BaseDeviceTensor(SingleDeviceTensor):
    pass


class DistributedTensor:

    def __init__(self, tensor_list: List[SingleDeviceTensor], base_device_idx: int):
        self._list = [BaseDeviceTensor(t) if i==base_device_idx else SingleDeviceTensor(t) for i,t in enumerate(tensor_list)]
        self._base_device_idx = base_device_idx
        self._lengths = torch.as_tensor([len(tensor) for tensor in tensor_list], dtype=torch.int64, device=self.base_device)
        # print('init:', self._lengths.device)

    def __len__(self):
        return torch.sum(self.lengths, dtype=torch.int64)

    def __getitem__(self, index):
        index_ = index.to(self.base_device) if isinstance(index, torch.Tensor) else index
        device_id_of_index = torch.searchsorted(self.offsets, index_, right=True) - 1
        unique_devices = torch.unique(device_id_of_index)
        if len(unique_devices)==1:
            part_device = self.parts[unique_devices].device
            part = self.parts[unique_devices]
            part_index = index_ - self.offsets[unique_devices]
            return part[part_index.to(part_device)]
        else:
            indices_per_part = [index[device_id_of_index==i]-o for i,o in enumerate(self.offsets)]
            return DistributedTensor([self.parts[i][idx] for i, idx in enumerate(indices_per_part)], base_device_idx=self.base_device_idx)

    def tolist(self):
        return self._list

    @property
    def offsets(self):
        return torch.cumsum(torch.as_tensor(self.lengths), 0, dtype=torch.int64) - self.lengths[0]

    @property
    def dtype(self):
        return self.parts[self.base_device_idx].dtype

    @property
    def lengths(self):
        return self._lengths

    @property
    def num_parts(self):
        return len(self._list)

    @property
    def base_device_idx(self):
        return self._base_device_idx

    @property
    def base_device(self):
        return self.parts[self.base_device_idx].device

    @property
    def parts(self):
        return self._list


    def __add__(self, tensor):
        assert_and_raise(tensor, DistributedTensor)
        return DistributedTensor([part + tensor.parts[i] for i, part in enumerate(self.parts)], base_device_idx=self.base_device_idx)

    def __sub__(self, tensor):
        assert_and_raise(tensor, DistributedTensor)
        return DistributedTensor([part - tensor.parts[i] for i, part in enumerate(self.parts)], base_device_idx=self.base_device_idx)

    def __mul__(self, scalar):
        assert_and_raise(scalar, float)
        return DistributedTensor([part * scalar for i, part in enumerate(self.parts)], base_device_idx=self.base_device_idx)

    def __pow__(self, exponent):
        assert_and_raise(exponent, int)
        return DistributedTensor([part ** exponent for i, part in enumerate(self.parts)], base_device_idx=self.base_device_idx)

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


class SummableDistributedTensor(DistributedTensor):
    pass
    

class BroadcastTensor(DistributedTensor):
    def at_device(self, device_idx):
        return self.parts[device_idx]

    def __repr__(self):
        return "BroadcastTensor\n" + str(self.parts[0]) 


class RowDistributedTensor(DistributedTensor):

    def __matmul__(self, tensor: BroadcastTensor):
        # TO DO: move this only to ColumnDistributedTensor
        assert_and_raise(tensor, BroadcastTensor)
        return RowDistributedTensor([part @ tensor.parts[i] for i, part in enumerate(self.parts)], base_device_idx=self.base_device_idx)

    @property
    def T(self):
        return ColumnDistributedTensor([part.T for part in self.parts], base_device_idx=self.base_device_idx)

    def zeros(row_sizes: Union[List, torch.Tensor], 
            num_columns: int, 
            dtype: torch.dtype, 
            device_list: List, 
            base_device_idx: int):
            return RowDistributedTensor([
                        torch.zeros(r, num_columns, dtype=dtype, device=d)
                        for r, d in zip(row_sizes, device_list)
                    ], base_device_idx=base_device_idx)


class ColumnDistributedTensor(DistributedTensor):

    def __matmul__(self, tensor: RowDistributedTensor):
        # TO DO: move this only to ColumnDistributedTensor
        assert_and_raise(tensor, RowDistributedTensor)
        return SummableDistributedTensor([part @ tensor.parts[i] for i, part in enumerate(self.parts)], base_device_idx=self.base_device_idx)
    
    @property
    def T(self):
        return RowDistributedTensor([part.T for part in self.parts], base_device_idx=self.base_device_idx)


if __name__ == "__main__":
    from eigenpro.device_manager import DeviceManager
    device_manager = DeviceManager([torch.device('cpu'), torch.device('cpu')], base_device_idx=0)
    tensor = DistributedTensor(torch.arange(20).reshape(-1,2).split(5), base_device_idx=1)
    
    # index = DistributedTensor([torch.LongTensor([7,8]), torch.LongTensor(112,13)])
    # print(tensor)
    index1 = SingleDeviceTensor(torch.LongTensor([2,3]))
    index2 = SingleDeviceTensor(torch.LongTensor([7,8]))
    index3 = BaseDeviceTensor(torch.LongTensor([4,2,9]))
    print(tensor[index1])
    print(tensor[index2])
    print(tensor[index3])

    zero = RowDistributedTensor.zeros(tensor.lengths, 2, tensor.dtype, device_manager.devices, device_manager.base_device_idx)

    # print(zero)
