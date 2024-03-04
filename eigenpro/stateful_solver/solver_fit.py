import numpy as np
from tabulate import tabulate
from termcolor import colored
from tqdm import tqdm
import torch
import torch.utils.data as torch_data

import eigenpro.data.array_dataset as array_dataset
import eigenpro.kernels as kernels
import eigenpro.preconditioners as pcd
import eigenpro.stateful_solver.solver as sol
import eigenpro.utils.mapreduce as mapreduce
import eigenpro.utils.metrics as metrics
import eigenpro.stateful_solver

def fit(model, X, Y, x, y, device, dtype=torch.float32, kernel=None,
        data_preconditioner_size=3_000, data_preconditioner_level=150, 
        model_preconditioner_size=3_000, model_preconditioner_level=150, 
        tmp_centers_coeff=2, wandb=None, T=None, epochs=1,
        accumulated_gradients=True):
    """Fit a kernel model using EigenPro method.

    Args:
        Z (torch.Tensor): centers. input tensor of shape [n_centers, n_features].
        X (torch.Tensor): training samples. input tensor of shape [n_samples, n_features].
        Y (torch.Tensor): labels for training samples. input tensor of shape [n_samples, n_classes].
        x (torch.Tensor): validation samples. input tensor of shape [n_valid_samples, n_features].
        y (torch.Tensor): labels for validation samples. input tensor of shape [n_valid_samples, n_classes].
        device(object): device object from device.py.
        dtype(torch.float32 or torch.float16): to save memory the default is torch.float16.
        kernel(function): kernel function (default is laplacian kernel with bandwidth 20.0)
        s_data(int): number of Nystrom samples for data preconditioner
        s_model(int): number of Nystrom samples for model preconditioner
        q_data(int): number of suppressed eigenvalues for data preconditioner
        q_model(int): number of suppressed eigenvalues for model preconditioner
        tmp_centers_coeff(int): ratio between total number of centers(temporary + Z) and number of original centers(Z)
        wandb(object): wandb object to log the result to wandb
        T(int): number of step to add temporary centers, note that if this is set to a number 'tmp_centers_coeff'
                will be ignored.
        epochs(int): number of epochs to run over the training samples
        accumulated_gradients: It should be true if Z and X are different, but if they are the same set this to False
                               for faster convergence.

    Returns:
        model(object): the trained model will be returned
    """


    device_base = device.device_base
    if kernel is None:
        kernel_fn = lambda x, z: kernels.laplacian(x, z, bandwidth= 20.0)
    else:
        kernel_fn = kernel

    # preconditioner
    nys_model_ids = np.random.choice(model.size, model_preconditioner_size, replace=False)
    nys_model = model.centers[nys_model_ids] 

    nys_data_ids = np.random.choice(len(X), data_preconditioner_size, replace=False)
    nys_data = X[nys_data_ids, :].to(device_base)

    data_preconditioner = pcd.Preconditioner(kernel_fn, nys_data, data_preconditioner_level)
    model_preconditioner = pcd.Preconditioner(kernel_fn, nys_model, model_preconditioner_level, nys_model_ids)
                   
    data_preconditioner.change_type(dtype=dtype)
    model_preconditioner.change_type(dtype=dtype)
                   
    # data loader
    dataset = array_dataset.ArrayDataset(X, Y)
    data_batch_size = min(8192, data_preconditioner.critical_batch_size)
    model_batch_size = min(8192, model_preconditioner.critical_batch_size)
    train_dataloader = torch_data.DataLoader(dataset, batch_size=data_batch_size , shuffle=True)

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

    model.eval()
    loss_test, accu_test = metrics.get_performance(model, x, y)
    # Print epoch summary using tabulate
    epoch_summary = [
        ["Test Loss", f"{loss_test:.10f}"],
        ["Test Accuracy", f"{accu_test * 100:.2f}%"],
    ]
    print(tabulate(epoch_summary, headers=[f"Epoch {epochs} Summary", "Value"], tablefmt="fancy_grid"))


    return model
