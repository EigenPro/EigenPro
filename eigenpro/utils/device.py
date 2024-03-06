import torch
import math
import torch.cuda.comm as torch_comm
from typing import List, Union
from termcolor import colored
from eigenpro.utils.tensor import (
        DistributedTensor, SingleDeviceTensor
    )


class DeviceManager:
    """Handles tensor operations across multiple devices."""

    def __init__(self, 
        use_gpu_if_available=True, 
        base_device_idx=0,
        load_weights=None):
        """Initializes the Device object with available devices."""
        
        self._base_device_idx = base_device_idx

        if torch.cuda.is_available() and use_gpu_if_available:
            self._devices = [torch.device(f'cuda:{i}')  for i in range(torch.cuda.device_count())]
            self.base_device = self.devices[base_device_idx]
        else:
            self._devices = [torch.device('cpu')]
            self.base_device = self.devices[0]

        self.is_multi_device = (len(self.devices) > 1) # True if
        

        print(colored(f'notice: the current implementation can only support 1 '
                      f'GPU, we only use the following device:'
                      f' ({self.base_device}) ','red'))

    @property
    def base_device_idx(self):
        return self._base_device_idx
    
    def chunk_sizes(self, size):
        # assumes equal load on all devices except base.
        # TO DO: a load-balancer for heterogenous multi-GPU systems

        def device_load(size, device_id):
            if device_id==self.base_device_idx:
                return math.floor(size/len(self.devices)) + math.remainder(size, len(self.devices))
            else:
                return math.floor(size/len(self.devices))

        return [device_load(size, device_id) for device_id in enumerate(self.devices)]


    @property
    def devices(self):
        return self._devices


    def broadcast(self, tensor: SingleDeviceTensor):
        if self.is_multi_device:
            return DistributedTensor(torch_comm.broadcast(tensor, self.devices))
        else:
            return SingleDeviceTensor(tensor)

    def scatter(self, tensor: SingleDeviceTensor):
        assert (isinstance(tensor, torch.Tensor) or isinstance(tensor, SingleDeviceTensor))
        if self.is_multi_device:
            return DistributedTensor(
                torch_comm.scatter(
                    tensor, self.devices, 
                    chunk_sizes=self.chunk_sizes(len(tensor))
                    )
                )
        else:
            return SingleDeviceTensor(tensor)

    def reduce_add(self, addable_tensor: Union[SingleDeviceTensor, DistributedTensor]) -> SingleDeviceTensor:
        if self.is_multi_device:
            return SingleDeviceTensor(torch_comm.reduce_add(addable_tensor, destination=self.base_device))
        else:
            return SingleDeviceTensor(addable_tensor)

    def gather(self, gatherable_tensor: Union[SingleDeviceTensor, DistributedTensor]) -> SingleDeviceTensor:
        if self.is_multi_device:
            return SingleDeviceTensor(torch_comm.gather(gatherable_tensor, destination=self.base_device))
        else:
            return SingleDeviceTensor(gatherable_tensor)