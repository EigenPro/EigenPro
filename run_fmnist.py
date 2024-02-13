import multiprocessing

import numpy as np
import torch

import eigenpro.data.utils as data_utils
from eigenpro.kernels import laplacian
import eigenpro.models.sharded_kernel_machine as skm
from eigenpro.run import run_eigenpro
from eigenpro.utils.cmd import parse_cmd_args
from eigenpro.utils.device import Device


def main():
    args = parse_cmd_args()

    X_train, X_test, Y_train, Y_test = data_utils.load_fmnist(
        "./data/fmnist", args.n_train, args.n_test)

    kernel_fn = lambda x, z: laplacian(x, z, bandwidth=20.)
    device = Device.create(use_gpu_if_available=True)
    
    # To run on CPU, dtype can not be `torch.float16` since
    # PyTorch does not support half-precision multiplication on CPU.
    if device.devices[0].type == 'cpu':
        dtype = torch.float32
    elif device.devices[0].type == 'cuda':
        dtype = torch.float16
    else:
        raise ValueError(f"Unknown device type: {device.devices[0].type}")

    # Note: if you want to use the whole X as your centers switch to
    # EigenPro 2.0 which is a faster method
    if args.model_size == -1:
        accumulated_gradients = False
        Z = X_train
    else:
    # In case you want to use a subset of data as model centers, define Z as
    # tensor of your centers
        accumulated_gradients = True
        centers_set_indices = np.random.choice(args.n_train, args.model_size,
                                               replace=False)
        Z = X_train[centers_set_indices,:]

    model = skm.create_sharded_kernel_machine(
        Z, Y_train.shape[-1], kernel_fn, device, dtype=dtype,
        tmp_centers_coeff=2)

    model = run_eigenpro(model, X_train, Y_train, X_test, Y_test, device,
                         dtype=dtype, kernel=kernel_fn, s_data=args.s_data,
                         s_model=args.s_model, q_data=args.q_data,
                         q_model=args.q_model, wandb=None, epochs=args.epochs,
                         accumulated_gradients=accumulated_gradients)

if __name__ == '__main__':
    # Call freeze_support() at the very beginning of the
    # if __name__ == '__main__' block.
    # This is required for execution in Windows.
    multiprocessing.freeze_support()
    main()
