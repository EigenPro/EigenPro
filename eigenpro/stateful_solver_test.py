import argparse
from enum import Enum, auto
import multiprocessing

import numpy as np
import torch

import eigenpro.data.utils as data_utils
import eigenpro.kernels as kernels
import eigenpro.models.sharded_kernel_machine as skm
import eigenpro.solver as solver
import eigenpro.utils.device as dev



def main():
    n, n_test, p, d, c = 1024, 128, 512, 16, 8
    epochs = 2
    s_data, s_model, q_data, q_model = 128, 64, 16, 8

    X_train = torch.randn(n, d)
    X_test = torch.randn(n_test, d)
    Y_train = torch.randn(n, c)
    Y_test = torch.randn(n_test, c)
    Z = torch.randn(p, d)

    kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.)
    device = dev.Device.create(use_gpu_if_available=True)

    # To run on CPU, dtype can not be `torch.float16` since
    # PyTorch does not support half-precision multiplication on CPU.
    if device.devices[0].type == 'cpu':
        dtype = torch.float32
    elif device.devices[0].type == 'cuda':
        dtype = torch.float16
    else:
        raise ValueError(f"Unknown device type: {device.devices[0].type}")

    model = km.KernelMachine(kernel_fn, d, c, p)
    model.centers = Z
    model.train()

    model = solver.fit(model, X_train, Y_train, X_test, Y_test, device,
                       dtype=dtype, kernel=kernel_fn,
                       s_data=s_data, s_model=s_model,
                       q_data=q_data, q_model=q_model,
                       wandb=None, epochs=epochs,
                       accumulated_gradients=True)

if __name__ == '__main__':
    # Call freeze_support() at the very beginning of the
    # if __name__ == '__main__' block.
    # This is required for execution in Windows.
    multiprocessing.freeze_support()
    main()
