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


class Dataset(Enum):
    FMNIST = auto()


def dataset_type(dataset_name):
    try:
        return Dataset[dataset_name.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(f"{dataset_name} is not a valid dataset name.")


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_train", type=int, help="Number of train samples", default=50_000
    )
    parser.add_argument(
        "--n_test", type=int, help="Number of test samples", default=10_000
    )
    parser.add_argument(
        "--model_size",
        type=int,
        help="Model size. Set to -1 to use the entire training dataset as model centers",
        default=20_000,
    )
    parser.add_argument(
        "--s_data",
        type=int,
        help="Number of Nystrom samples for Data Preconditioner",
        default=2_000,
    )
    parser.add_argument(
        "--q_data", type=int, help="Level of Data Preconditioner", default=100
    )
    parser.add_argument(
        "--s_model",
        type=int,
        help="Number of Nystrom samples for Model Preconditioner",
        default=2_000,
    )
    parser.add_argument(
        "--q_model", type=int, help="Level of Model Preconditioner", default=100
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=2
    )
    parser.add_argument(
        "--dataset",
        type=dataset_type,
        help="Name of dataset. Currectly support `fmnist`.",
    )
    parser.add_argument(
        "--data_path", type=str, help="Path of the dataset.", default=None
    )
    return parser.parse_args()


def main():
    args = parse_cmd_args()

    data_path = args.data_path
    if args.dataset == Dataset.FMNIST:
        if data_path is None:
            data_path = "./__data__/fmnist"
        X_train, X_test, Y_train, Y_test = data_utils.load_fmnist(
            data_path, args.n_train, args.n_test
        )

    kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth=20.0)
    device = dev.Device.create(use_gpu_if_available=True)

    # To run on CPU, dtype can not be `torch.float16` since
    # PyTorch does not support half-precision multiplication on CPU.
    if device.devices[0].type == "cpu":
        dtype = torch.float32
    elif device.devices[0].type == "cuda":
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
        centers_set_indices = np.random.choice(
            args.n_train, args.model_size, replace=False
        )
        Z = X_train[centers_set_indices, :]

    model = skm.create_sharded_kernel_machine(
        Z, Y_train.shape[-1], kernel_fn, device, dtype=dtype, tmp_centers_coeff=2
    )

    model = solver.fit(
        model,
        X_train,
        Y_train,
        X_test,
        Y_test,
        device,
        dtype=dtype,
        kernel=kernel_fn,
        s_data=args.s_data,
        s_model=args.s_model,
        q_data=args.q_data,
        q_model=args.q_model,
        wandb=None,
        epochs=args.epochs,
        accumulated_gradients=accumulated_gradients,
    )


if __name__ == "__main__":
    # Call freeze_support() at the very beginning of the
    # if __name__ == '__main__' block.
    # This is required for execution in Windows.
    multiprocessing.freeze_support()
    main()
