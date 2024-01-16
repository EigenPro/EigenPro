import time
import numpy as np
from .kernels import laplacian
from .preconditioner import Preconditioner
from .data import ArrayDataset
from .optimizers import EigenPro
import torch
from torch.utils.data import DataLoader
from .utils import MapReduceEngein
from .models import create_kernel_model
import ipdb
from termcolor import colored
from tabulate import tabulate
from tqdm import tqdm


def get_performance(model, X, Y, batch_size=1024):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure model is in evaluation mode

    # Convert X and Y to PyTorch datasets and use DataLoader for batch processing
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    total_loss = 0
    total_accuracy = 0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_Y in dataloader:
            # Move batch to GPU
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            # Forward pass
            Y_hat = model(batch_X).to(device)

            # Calculate loss and accuracy
            loss = torch.norm(Y_hat - batch_Y)**2 / batch_Y.size(0)
            accuracy = torch.sum(torch.argmax(Y_hat, dim=1) == torch.argmax(batch_Y, dim=1)).item()

            
            # Accumulate loss and accuracy
            total_loss += loss.item() * batch_Y.size(0)
            total_accuracy += accuracy
            total_samples += batch_Y.size(0)
            
            del batch_X, batch_Y

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples

    return avg_loss, avg_accuracy

def run_eigenpro(Z, X, Y, x, y, device, type=torch.float32, kernel=None,
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
        type(torch.float32 or torch.float16): to save memory the default is torch.float16.
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


    p = Z.shape[0]
    n = X.shape[0]
    d_out = Y.shape[-1]
    device_base = device.device_base
    if kernel is None:
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


    # data loader
    dataset = ArrayDataset(X, Y)

    batch_size = precon_data.critical_batch_size
    train_dataloader = DataLoader(dataset, batch_size=batch_size , shuffle=True)

    # model initilization
    model = create_kernel_model(Z,d_out,kernel_fn,device,type=type,tmp_centers_coeff = tmp_centers_coeff)

    # optimizer
    optimizer = EigenPro(model, p, precon_data,precon_model,kz_xs_evecs,type,
                         accumulated_gradients=accumulated_gradients)
    # projection frequency
    if T is None:
        T = ((tmp_centers_coeff-1)*p-s_data)//precon_data.critical_batch_size    #2

    # configuration summary
    data = [
        [colored("Training size (n)", 'green'), X.shape[0]],
        [colored("Number of centers (p)", 'green'), Z.shape[0]],
        [colored("Ambient dimension (d)", 'green'), Z.shape[1]],
        [colored("output dimension", 'green'), Y.shape[1]],
        [colored("# of Nystrom samples (data preconditioner)", 'green'), s_data],
        [colored("# suppressed eigenvalues (data preconditioner)", 'green'), q_data],
        [colored("# of Nystrom samples (model preconditioner)", 'green'), s_model],
        [colored("# suppressed eigenvalues (model preconditioner)", 'green'), q_model],
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
                           f'q_model': f'{q_model}',
                           })

    project_counter = 0
    for epoch in range(epochs):
        epoch_progress = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                              desc=f"Epoch {epoch + 1}/{epochs}")

        for t,(x_batch,y_batch,id_batch) in epoch_progress:

            optimizer.step(x_batch, y_batch, id_batch)
            if torch.cuda.is_available():
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
