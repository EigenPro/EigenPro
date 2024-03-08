import numpy as np
from tabulate import tabulate
from termcolor import colored
from tqdm import tqdm
import torch
import torch.utils.data as torch_data

from eigenpro.utils.tensor import BaseDeviceTensor

import eigenpro.data.array_dataset as array_dataset
import eigenpro.kernels as kernels
import eigenpro.preconditioners as pcd
import eigenpro.solver as sol
import eigenpro.utils.mapreduce as mapreduce
import eigenpro.utils.metrics as metrics


def fit(model, 
    X, Y, x=None, y=None, dtype=torch.float32, kernel=None,
    data_preconditioner_size=3_000, data_preconditioner_level=150, 
    model_preconditioner_size=3_000, model_preconditioner_level=150, 
    tmp_centers_coeff=2, wandb=None, T=None, epochs=1,
    accumulated_gradients=True):
    """Fit a kernel model using EigenPro method.
    Returns:
        model(object): the trained model will be returned
    """
    device_manager = model.device_manager
    base_device = device_manager.base_device
    base_device_idx = device_manager.base_device_idx

    if kernel is None:
        kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth= 20.0)
    else:
        kernel_fn = kernel

    # preconditioner
    nys_model_ids = model.device_offsets[base_device_idx] + BaseDeviceTensor( # make sure all model nystrom centers are on base device 
        torch.randperm(
            device_manager.chunk_sizes(model.size)[base_device_idx], 
            generator=torch.Generator(device=base_device))[:model_preconditioner_size]
        )

    nys_model = model.centers[nys_model_ids]

    nys_data_ids = torch.randperm(len(X), generator=torch.Generator(device=base_device))[:data_preconditioner_size]
    nys_data = BaseDeviceTensor(X[nys_data_ids.to(X.device), :].to(base_device))

    data_preconditioner = pcd.Preconditioner(kernel_fn, nys_data, data_preconditioner_level)
    model_preconditioner = pcd.Preconditioner(kernel_fn, nys_model, model_preconditioner_level, nys_model_ids)

    # data_preconditioner.change_type(dtype=dtype)
    # model_preconditioner.change_type(dtype=dtype)
                   
    # data loader
    dataset = array_dataset.ArrayDataset(X, Y)
    data_batch_size = min(8192, data_preconditioner.critical_batch_size)
    model_batch_size = min(8192, model_preconditioner.critical_batch_size)
    train_dataloader = torch_data.DataLoader(
        dataset, batch_size=data_batch_size , shuffle=True)

    eigenpro_solver = sol.EigenProSolver(
        model, 
        data_preconditioner,
        model_preconditioner,
        dtype,
        tmp_centers_coeff,
        epochs_per_projection = 1)

    # projection frequency
    if T is None:
        T = ((tmp_centers_coeff-1)*model.size)//data_preconditioner.critical_batch_size    #2
    
    # configuration summary
    data = [
        [colored("size of training dataset", 'green'), X.shape[0]],
        [colored("critical batch size (SGD)",'green'), data_preconditioner.critical_batch_size],
        [colored("batch size (SGD)",'green'), data_batch_size],
        [colored("size of model", 'blue'), model.size],
        [colored("ambient dimension", 'blue'), X.shape[1]],
        [colored("output dimension", 'blue'), Y.shape[1]],
        [colored("size of data preconditioner", 'yellow'), data_preconditioner_size],
        [colored("level of data preconditioner", 'yellow'), data_preconditioner_level],
        [colored("size of model preconditioner", 'yellow'), model_preconditioner_size],
        [colored("level of model preconditioner", 'yellow'), model_preconditioner_level],
        [colored("critical batch size (projection)",'red'), model_preconditioner.critical_batch_size],
        [colored("batch size (projection)", 'red'), model_batch_size],
        [colored("scaled learning rate",'red'), 
        f"{data_preconditioner.scaled_learning_rate(data_preconditioner.critical_batch_size):.2f}"],
        [colored("projection interval (in batches)",'red'), T]
    ]
    # Table Formatting
    table = tabulate(data, headers=["Configuration", "Value"], tablefmt="fancy_grid",
                     numalign="middle")
    # Print the table
    print(table)

    eigenpro_solver.fit(train_dataloader, epochs)

    # calculate test error if test data provided
    if (x is not None) and (y is not None):
        
        loss_test, accu_test = metrics.get_performance(model, x, y)
        # Print epoch summary using tabulate
        epoch_summary = [
            ["Test Loss", f"{loss_test:.10f}"],
            ["Test Accuracy", f"{accu_test * 100:.2f}%"],
        ]
        print(tabulate(epoch_summary, headers=[f"Epoch {epochs} Summary", "Value"], tablefmt="fancy_grid"))


    return model
