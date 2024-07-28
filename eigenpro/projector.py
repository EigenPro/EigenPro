import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as torch_data
import eigenpro.preconditioner as pcd
import eigenpro.data.array_dataset as array_dataset




def project(model, kernel_fn,s,q,device,labels=None, preconditioner = None, update_preconditioner = False):
    """Fit a kernel model using EigenPro method.

    ### ATTENTION: WORKS ONLY WITH 1-GPU!

     Args:
         model (KernelMachine): A KernelMachine instance.
         device(object): device object from device.py.
         kernel(function): kernel function (default is laplacian kernel with bandwidth 20.0)
         s(int): number of Nystrom samples for model preconditioner
         q(int): number of suppressed eigenvalues for model preconditioner
         preconditioner: tuple (model_preconditioner, nys_model_indices) if None they will be created.
                        Preconditioner instance that contains a top kernel eigensystem for correcting the gradient for
                        the projection.
         update_preconditioner: if preconditioner is not None but we want to recalculate the preconditioner.

     Returns:
         model(object): the trained model will be returned
     """


    p = model.centers.shape[0]
    dtype = model.centers.dtype

    device_base = device.device_base

    model_preconditioner, nys_model_indices = preconditioner

    if model_preconditioner is None or update_preconditioner:
        nys_model_indices = np.random.choice(p, s, replace=False)
        nys = model.centers[nys_model_indices, :].to(device_base)
        model_preconditioner = pcd.Preconditioner(kernel_fn, nys, q)
        model_preconditioner.change_type(dtype=dtype)



    model_batch_size = min(10_000, model_preconditioner.critical_batch_size)



    # make data loader
    if labels is None:
        print("Evaluating labels for projection ...")
        raise NotImplementedError("This method should be overridden by subclasses.")


    # ipdb.set_trace()
    projection_dataset = array_dataset.ArrayDataset(model.centers, labels)
    projection_loader = torch_data.DataLoader(
        projection_dataset, batch_size=model_batch_size, shuffle=True)

    d_out = labels.shape[-1]
    #wiegths
    weights_project = torch.zeros(p, d_out,
                                  device=device_base, dtype = dtype)

    # ipdb.set_trace()
    for _ in range(1):
        for z_batch, labels_batch, id_batch in tqdm(
                projection_loader,
                total=len(projection_loader),
                leave=False,
                desc="Projection"
        ):
            # optimizer.step(z_batch, grad_batch, id_batch, projection=True)

            kernel_mat = kernel_fn(z_batch, model.centers)
            batch_p = kernel_mat@weights_project

            grad = batch_p - labels_batch.to(dtype).to(batch_p.device)
            batch_size = z_batch.shape[0]

            lr = model_preconditioner.scaled_learning_rate(batch_size)
            deltap, delta = model_preconditioner.delta(
                z_batch.to(grad.device).to(dtype), grad)

            weights_project[id_batch,:] =  weights_project[id_batch,:]-lr*grad


            weights_project[nys_model_indices,:] = weights_project[nys_model_indices,:] + lr * delta





    return weights_project, (model_preconditioner,nys_model_indices)

