import time
import numpy as np
from kernels import laplacian
from preconditioner import Preconditioner
from data import ArrayDataset
from optimizers import EigenPro
import torch
from torch.utils.data import DataLoader
from utils import create_kernel_model, MapReduceEngein
import ipdb
from termcolor import colored
from tabulate import tabulate
from tqdm import tqdm

def get_perform(model, X,Y):
    Y_hat = model.forward(X, train=False)
    loss_test = torch.norm(Y_hat.to('cpu') - Y) / len(Y)
    accu_test = sum(torch.argmax(Y_hat.to('cpu'), dim=1) == torch.argmax(Y, dim=1)) / len(Y)
    return loss_test, accu_test

def run_eigenpro(Z, X, Y, x, y, device,type=torch.float32, kernel=None,
                 s_data= 3_000, s_model= 3_000, q_data=150, q_model=150,
                 tmp_centers_coeff = 2, wandb = None, T =None, epochs=1, accumulated_gradients = True):

    """wrapper to run eigenpro
    Args:
        Z (torch.Tensor): centers. input tensor of shape [n_centers, n_features].
        X (torch.Tensor): training samples. input tensor of shape [n_samples, n_features].
        Y (torch.Tensor): labels for training samples. input tensor of shape [n_samples, n_classes].
        x (torch.Tensor): validation samples. input tensor of shape [n_valid_samples, n_features].
        y (torch.Tensor): labels for validation samples. input tensor of shape [n_valid_samples, n_classes].
        device(object): device object from device.py.
        type(torch.flaot32 or torch.flaot16): to save memory the default is torch.float16.
        kernel(function): kernel function (default is laplacian kernel with bandwidth 20.0)
        s_data(int): number of Nystrom samples for data preconditioner
        s_model(int): number of Nystrom samples for model preconditioner
        q_data(int): number of suppressed eigenvalues for data preconditioner
        q_model(int): number of suppressed eigenvalues for model preconditioner
        tmp_centers_coeff(int): ratio between total number of centers(temporary + Z) and number of original centers(Z)
        wandb(object): wnadb object to log the result to wandb
        T(int): number of step to add temporary centers, note that if this is set to a number 'tmp_centers_coeff'
                will be ignored.
        epochs(int): number of epochs to run over the training smaples
        accumulated_gradients: It should be true if Z and X are different, but if they are the same set this to False
                               for faster convergence.


    Returns:
        model(object): the traiend model will be returned
    """


    p = Z.shape[0]
    n = X.shape[0]
    d_out = Y.shape[-1]
    device_base = device.device_base
    if kernel == None:
        kernel_fn = lambda x, z: laplacian(x, z, bandwidth= 20.0)
    else:
        kernel_fn = kernel



    #### preconditioner
    nys_model = Z[0:s_model] # we are assuming first s_model of Z are being used as Nystrom
                             # samples for the projection problem
    nys_data_indices = np.random.choice(X.shape[0], s_data, replace=False)
    nys_data = X[nys_data_indices, :]

    precon_data = Preconditioner(kernel_fn, nys_data, q_data)
    precon_model = Preconditioner(kernel_fn, nys_model, q_model)
    kz_xs_evecs = precon_data.eval_vec(Z).to(device_base).to(type)
    precon_data.change_type(type=type)
    precon_model.change_type(type=type)


    ##### data loader
    dataset = ArrayDataset(X, Y)

    batch_size = precon_data.critical_batch_size
    train_dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True)

    ###### model initilization
    model = create_kernel_model(Z,d_out,kernel_fn,device,type=type,tmp_centers_coeff = tmp_centers_coeff)
    #### optimizer

    optimizer = EigenPro(model, p, precon_data,precon_model,kz_xs_evecs,type,
                         accumulated_gradients=accumulated_gradients)
    ### projection frequency
    if T is None:
        T = ((tmp_centers_coeff-1)*p-s_data)//precon_data.critical_batch_size    #2

    ####### configs summary
    data = [
        [colored("Training size (n)", 'green'), X.shape[0]],
        [colored("Number of centers (p)", 'green'), Z.shape[0]],
        [colored("Ambient dimension (d)", 'green'), Z.shape[1]],
        [colored("output dimension", 'green'), Y.shape[1]],
        [colored("# of Nystrom samples (data precond.)", 'green'), s_data],
        [colored("# supressed eigenvalues (data precond.)", 'green'), q_data],
        [colored("# of Nystrom samples (model precond.)", 'green'), s_model],
        [colored("# supressed eigenvalues (model precond.)", 'green'), q_model],
        [colored("Batch Size (Critical)",'green'), precon_data.critical_batch_size],
        [colored("Scaled Learning Rate",'green'),
         f"{precon_data.scaled_learning_rate(precon_data.critical_batch_size):.2f}"],
        [colored("Projection Frequency (T)",'green'), T]
    ]
    # Table Formatting
    table = tabulate(data, headers=["Configuration", "Value"], tablefmt="fancy_grid",
                     numalign="middle")
    # Print the table
    print(table)

    if wandb is not None:
        wandb.config.update({f'Project Frequency':f'{T}',
                           f'batch_size':f'{batch_size}',
                           f'number of training samples': f'{n}',
                           f'number of centers': f'{p}',
                           f's_data':f'{s_data}',
                           f'q_data': f'{q_data}',
                           f's_model': f'{s_model}',
                           f's_model': f'{s_model}',
                           })

    project_counter = 0
    for epoch in range(epochs):
        epoch_progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                              desc=f"Epoch {epoch + 1}/{epochs}")

        for t,(x_batch,y_batch,id_batch) in epoch_progress:

            optimizer.step(x_batch, y_batch, id_batch)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


            if ( (project_counter + 1) % T == 0 or (t==len(train_dataloader)-1) ) and accumulated_gradients:
                projection_dataset = ArrayDataset(Z, optimizer.grad_accumulation)
                projection_loader = DataLoader(projection_dataset,
                                               batch_size=precon_model.critical_batch_size, shuffle=True)
                for _ in range(1):
                    for z_batch, grad_batch, id_batch in projection_loader:
                        optimizer.step(z_batch, grad_batch, id_batch, projection=True)

                update_projection = torch.cat([k.weights_project.to(device_base) for k in model.shard_kms])

                model.update_by_index(torch.tensor(list(range(p))), update_projection)
                model.reset()
                optimizer.reset()
                torch.cuda.synchronize()


            if wandb is not None:
                loss_test, accu_test = get_perform(model, x, y)
                wandb.log({ 'test loss': loss_test.item(),
                            'test accuracy': accu_test})

            project_counter += 1

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        loss_test, accu_test = get_perform(model, x, y)
        # Print epoch summary using tabulate
        epoch_summary = [
            ["Test Loss", f"{loss_test:.10f}"],
            ["Test Accuracy", f"{accu_test * 100:.2f}%"],
        ]
        print(tabulate(epoch_summary, headers=[f"Epoch {epoch + 1} Summary", "Value"], tablefmt="fancy_grid"))


    return model
