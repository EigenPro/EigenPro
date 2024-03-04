import argparse
from enum import Enum, auto
import multiprocessing

import numpy as np
import torch

import eigenpro.data.utils as data_utils
import eigenpro.kernels as kernels
import eigenpro.models.kernel_machine as km
from eigenpro.stateful_solver.solver_fit import fit
import eigenpro.utils.device as dev



def main():
    n_train, n_test, model_size, d_in, d_out = 16384, 128, 4096*3, 16, 8
    epochs = 4
    data_preconditioner_size, data_preconditioner_level = 128, 16
    model_preconditioner_size, model_preconditioner_level = 64, 8

    X_train = torch.randn(n_train, d_in)
    X_test = torch.randn(n_test, d_in)
    Y_train = torch.randn(n_train, d_out)
    Y_test = torch.randn(n_test, d_out)
    Z = torch.randn(model_size, d_in)

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

    model = km.KernelMachine(kernel_fn, d_in, d_out, model_size)
    model.centers = Z
    model.train()

    model = fit(
        model, 
        X_train, Y_train, X_test, Y_test, device,
        dtype=dtype, kernel=kernel_fn,
        data_preconditioner_size=data_preconditioner_size, 
        data_preconditioner_level=data_preconditioner_level, 
        model_preconditioner_size=model_preconditioner_size,
        model_preconditioner_level=model_preconditioner_level,
        wandb=None, epochs=epochs,
    )

if __name__ == '__main__':
    # Call freeze_support() at the very beginning of the
    # if __name__ == '__main__' block.
    # This is required for execution in Windows.
    multiprocessing.freeze_support()
    main()
