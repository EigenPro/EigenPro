import argparse
from enum import Enum, auto
import multiprocessing

import numpy as np
import torch

import os 
data_dir = os.environ['DATA_DIR']

import eigenpro.data.utils as data_utils
import eigenpro.kernels as kernels
import eigenpro.models.kernel_machine as km
from eigenpro.stateful_solver.solver_fit import fit
import torch.nn.functional as F
import eigenpro.utils.device as dev

import eigenpro.data.utils as data_utils

def main():
    n_train, n_test, model_size = 50000, 10000, 20000
    epochs = 2
    data_preconditioner_size, data_preconditioner_level = 2000, 100
    model_preconditioner_size, model_preconditioner_level = 2000, 100

    X_train, X_test, Y_train, Y_test = data_utils.load_fmnist(data_dir, n_train, n_test) 
 
    d_in, d_out = X_train.shape[1], Y_train.shape[1]
    Z = X_train[torch.randperm(len(X_train))[:model_size]]

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

    model = km.KernelMachine(
        kernel_fn, d_in, d_out, model_size, centers=Z, device=device.devices[0], dtype=dtype)

    model = fit(
        model, 
        X_train[:n_train], Y_train[:n_train], X_test, Y_test, device,
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
