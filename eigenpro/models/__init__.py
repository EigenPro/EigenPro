"""Classes of Kernel Machines."""

from typing import Callable, List, Optional
import torch
from ..utils.device import Device
from .sharded import ShardedKernelMachine
from .preallocated import PreallocatedKernelMachine_optimized


def create_kernel_model(centers, n_outputs, kernel_fn,device, dtype=torch.float32, tmp_centers_coeff=2):

    # ipdb.set_trace()
    list_of_centers = [centers] #device(centers, strategy="divide_to_gpu")
    kms = []
    for i, centers_i in enumerate(list_of_centers):
        kms.append(
            PreallocatedKernelMachine_optimized(
                kernel_fn, 
                n_outputs, 
                centers_i, 
                dtype=dtype, 
                device=device.devices[i],
                tmp_centers_coeff=tmp_centers_coeff
            )
        )

    del list_of_centers
    # ipdb.set_trace()
    return ShardedKernelMachine(kms, device)


