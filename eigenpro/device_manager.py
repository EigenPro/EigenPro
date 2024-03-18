import torch
import math
import torch.cuda.comm as torch_comm
# import eigenpro.utils.ops as torch_comm
from typing import List, Union
from termcolor import colored
from eigenpro.utils.tensor import (
        BroadcastTensor, RowDistributedTensor, SummableDistributedTensor,
        SingleDeviceTensor, BaseDeviceTensor, 
    )


class DeviceManager:

    def __init__(self, 
            device_list = None,
            use_gpu_if_available=True, 
            base_device_idx=0,
        ):
        
        self._base_device_idx = base_device_idx

        if device_list is None:
            if torch.cuda.is_available() and use_gpu_if_available:
                self._devices = [torch.device(f'cuda:{i}')  for i in range(torch.cuda.device_count())]
            else:
                self._devices = [torch.device('cpu')]    
        else:    
            self._devices = device_list

        self.is_multi_device = len(self.devices) > 1
        

    @property
    def base_device(self):
        return self.devices[self.base_device_idx]

    @property
    def base_device_idx(self):
        return self._base_device_idx
    
    def chunk_sizes(self, size):
        # assumes equal load on all devices except perhaps base.
        # TO DO: a load-balancer for heterogenous multi-GPU systems

        def device_load(size, device_id):
            if device_id==self.base_device_idx: # put remainder on base device
                return math.floor(size/len(self.devices)) + math.remainder(size, len(self.devices))
            else:
                return math.floor(size/len(self.devices))

        return torch.as_tensor([device_load(size, device_id) for device_id in enumerate(self.devices)])

    @property
    def devices(self):
        return self._devices

    def broadcast(self, tensor: SingleDeviceTensor):
        if self.is_multi_device:
            return BroadcastTensor(torch_comm.broadcast(tensor, self.devices), base_device_idx=self.base_device_idx)
        else:
            return BaseDeviceTensor(tensor.to(self.base_device))

    def scatter(self, tensor: SingleDeviceTensor):
        assert (isinstance(tensor, torch.Tensor) or isinstance(tensor, SingleDeviceTensor))
        if self.is_multi_device:
            return RowDistributedTensor(
                torch_comm.scatter(
                    tensor, devices=self.devices, 
                    chunk_sizes=self.chunk_sizes(len(tensor)).tolist()
                    ),
                base_device_idx=self.base_device_idx
                )
        else:
            return BaseDeviceTensor(tensor.to(self.base_device))

    def reduce_add(self, addable_tensor: Union[SingleDeviceTensor, SummableDistributedTensor]) -> BaseDeviceTensor:
        if self.is_multi_device:
            return BaseDeviceTensor(torch_comm.reduce_add(addable_tensor.tolist(), destination=self.base_device))
        else:
            return BaseDeviceTensor(addable_tensor.to(self.base_device))

    def gather(self, gatherable_tensor: Union[SingleDeviceTensor, RowDistributedTensor]) -> SingleDeviceTensor:
        if self.is_multi_device:
            return BaseDeviceTensor(torch_comm.gather(gatherable_tensor.tolist(), destination=self.base_device))
        else:
            return BaseDeviceTensor(gatherable_tensor)
