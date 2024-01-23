import time
import numpy as np
from .kernels import laplacian
from .preconditioner import Preconditioner
from .data.array_dataset import ArrayDataset
from .optimizers import EigenPro
import torch
from torch.utils.data import DataLoader
from .utils.mapreduce import MapReduceEngine
from .models import create_kernel_model
from termcolor import colored
from tabulate import tabulate
from tqdm import tqdm 
from .utils.metrics import get_performance

from .utils.ops import  choose_from_list

import ipdb

def run_eigenpro(model, X, Y, x, y, device, dtype=torch.float32, kernel=None,
                 s_data=3_000, s_model=3_000, q_data=150, q_model=150,
                 tmp_centers_coeff=2, wandb=None, T=None, epochs=1, accumulated_gradients=True):
    """wrapper to run eigenpro
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


    p = model.size 
    n = X.shape[0]
    device_base = device.device_base
    if kernel is None:
        kernel_fn = lambda x, z: laplacian(x, z, bandwidth= 20.0)
    else:
        kernel_fn = kernel



    # preconditioner
    # nys_model = model.centers[0][0:s_model] # we are assuming first s_model of Z are being used as Nystrom
    #                                         # samples for the projection problem

    nys_model = choose_from_list(model.centers, s_model)
    nys_model = torch.cat([nys.to(device_base) for nys in nys_model])


    nys_data_indices = np.random.choice(X.shape[0], s_data, replace=False)
    nys_data = X[nys_data_indices, :].to(device_base)

    data_preconditioner = Preconditioner(kernel_fn, nys_data, q_data)
    model_preconditioner = Preconditioner(kernel_fn, nys_model, q_model)
                   
    data_preconditioner.change_type(dtype=dtype)
    model_preconditioner.change_type(dtype=dtype)
                   
    # kz_xs_evecs = data_preconditioner.eval_vec(model.centers).to(device_base).type(dtype)
    kz_xs_evecs = data_preconditioner.eval_vec(model.centers,device_base).type(dtype)

    # data loader
    dataset = ArrayDataset(X, Y)
    data_batch_size = min(10_000, data_preconditioner.critical_batch_size)
    model_batch_size = min(10_000, model_preconditioner.critical_batch_size)
    train_dataloader = DataLoader(dataset, batch_size=data_batch_size , shuffle=True)

    # optimizer
    optimizer = EigenPro(model, p, data_preconditioner,model_preconditioner,kz_xs_evecs,dtype,
                         accumulated_gradients=accumulated_gradients)
    # projection frequency
    if T is None:
        T = ((tmp_centers_coeff-1)*p-s_data)//data_preconditioner.critical_batch_size    #2

    # configuration summary
    data = [
        [colored("size of model", 'green'), model.size],
        [colored("ambient dimension", 'green'), X.shape[1]],
        [colored("output dimension", 'green'), Y.shape[1]],
        [colored("size of data preconditioner", 'green'), s_data],
        [colored("level of data preconditioner", 'green'), q_data],
        [colored("size of model preconditioner", 'green'), s_model],
        [colored("level of model preconditioner", 'green'), q_model],
        [colored("size of training dataset", 'green'), X.shape[0]],
        [colored("critical batch size (SGD)",'green'), data_preconditioner.critical_batch_size],
        [colored("batch size (SGD)",'green'), data_batch_size],
        [colored("critical batch size (projection)",'green'), model_preconditioner.critical_batch_size],
        [colored("batch size (projection)", 'green'), model_batch_size],
        [colored("scaled learning rate",'green'),
         f"{data_preconditioner.scaled_learning_rate(data_preconditioner.critical_batch_size):.2f}"],
        [colored("projection interval (in batches)",'green'), T]
    ]
    # Table Formatting
    table = tabulate(data, headers=["Configuration", "Value"], tablefmt="fancy_grid",
                     numalign="middle")
    # Print the table
    print(table)

    if wandb is not None:
        wandb.config.update({f'Project Frequency':f'{T}',
                           f'batch_size':f'{data_batch_size}',
                           f'number of training samples': f'{n}',
                           f'number of centers': f'{p}',
                           f's_data':f'{s_data}',
                           f'q_data': f'{q_data}',
                           f's_model': f'{s_model}',
                           f'q_model': f'{q_model}',
                           })

    project_counter = 0
    for epoch in range(epochs):
        epoch_progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                              desc=f"Epoch {epoch + 1}/{epochs}")

        for t, (x_batch,y_batch,id_batch) in epoch_progress:

            optimizer.step(x_batch, y_batch, id_batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # if t%1 == 0:
            #     loss_test, accu_test = get_performance(model, x, y)
            #     # Print epoch summary using tabulate
            #     epoch_summary = [
            #         ["Test Loss", f"{loss_test:.10f}"],
            #         ["Test Accuracy", f"{accu_test * 100:.2f}%"],
            #     ]
            #     print(tabulate(epoch_summary, headers=[f"Epoch {epoch + 1}--iteration{t} Summary", "Value"], tablefmt="fancy_grid"))

            if ( (project_counter + 1) % T == 0 or (t==len(train_dataloader)-1) ) and accumulated_gradients:

                # for ind_g,g in enumerate(self.device.devices):
                #     projection_datasets = ArrayDataset(model.centers[ind_g], optimizer.grad_accumulation)
                #     projection_loader = DataLoader(projection_dataset,
                #                                    batch_size=model_batch_size, shuffle=True)

                grad_accumulation_parts = torch.chunk(optimizer.grad_accumulation, len(device.devices), dim=0)
                dataloaders = []

                for ind_g, g in enumerate(device.devices):
                    # Get the respective part of model.centers and grad_accumulation
                    centers_part = model.centers[ind_g]
                    grad_accumulation_part = grad_accumulation_parts[ind_g]

                    # Move grad_accumulation_part to the respective GPU
                    grad_accumulation_part = grad_accumulation_part.to(g)

                    # Create the dataset with centers_part and the corresponding grad_accumulation_part
                    dataset = ArrayDataset(centers_part, grad_accumulation_part)



                    # Create a DataLoader for the dataset
                    loader = DataLoader(dataset, batch_size=model_batch_size, shuffle=True)

                    # Store the DataLoader
                    dataloaders.append(loader)

                for _ in range(1):
                    for inddl,dl in enumerate(dataloaders):
                        for z_batch, grad_batch, id_batch in tqdm(
                            dl, total=len(dl), leave=False):
                            optimizer.step(z_batch, grad_batch, id_batch, projection=True,gpu_ind=inddl)

                update_projection = torch.cat([k.weights_project.to(device_base) for k in model.shard_kms])

                model.update_by_index(torch.tensor(list(range(p))), update_projection)
                model.reset()
                optimizer.reset()
                # print('checking used capacity:')
                # for m in model.shard_kms:
                #     print(f'used capacity:{m.used_capacity}')
                if torch.cuda.is_available():
                    torch.cuda.synchronize()


            if wandb is not None:
                loss_test, accu_test = get_performance(model, x, y)
                wandb.log({'test loss': loss_test.item(),
                           'test accuracy': accu_test})

            project_counter += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        loss_test, accu_test = get_performance(model, x, y)
        # Print epoch summary using tabulate
        epoch_summary = [
            ["Test Loss", f"{loss_test:.10f}"],
            ["Test Accuracy", f"{accu_test * 100:.2f}%"],
        ]
        print(tabulate(epoch_summary, headers=[f"Epoch {epoch + 1} Summary", "Value"], tablefmt="fancy_grid"))


    return model
