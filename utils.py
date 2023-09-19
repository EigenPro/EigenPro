""" Common utils for package EigenPro_v3.2
"""
import torch
from device import Device
from models import PreallocatedKernelMachine, ShardedKernelMachine
from collections import OrderedDict

DEFAULT_DTYPE = torch.float32



def create_kernelmodel(Z,n_outputs,kernel_fn):
    device = Device.create()
    Z_list = device(Z, strategy="divide_to_gpu")
    kms = []
    for i,zi in enumerate(Z_list):
        kms.append(ShardedKernelMachine( kernel_fn,n_outputs,zi,device=device.devices[i]) )

    return ShardedKernelMachine(kms,device)


class LRUCache:
    def __init__(self):
        self.cache = {}

    def get(self, key: int) -> int:
        return self.cache.get(key, -1)

    def put(self, key: int, value: int) -> None:
        self.cache.clear()  # Since capacity is 1, clear the cache before adding a new item
        self.cache[key] = value




