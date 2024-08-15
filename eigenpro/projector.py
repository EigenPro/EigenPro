import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as torch_data
import eigenpro.preconditioner as pcd
import eigenpro.data.array_dataset as array_dataset


def project(centers, target_evals, preconditioner, nys_centers, kernel_fn):
    """Project the target function into the space defined by kernel-induced basis functions using chosen centers

    # ATTENTION: WORKS ONLY WITH 1-GPU!

     Args:
         centers (Tensor): Data points used to span a kernel space and define a kernel machine.
         target_evals(Tensro): Evaluation of target function on the centers of the projection space.
         preconditioner: Preconditioner instance that contains a top kernel eigensystem for correcting
         the gradient forthe projection.
         nys_centers: indices of centers that used for preconditioner.
         kernel(function): kernel function.


     Returns:
         weights_project(Tensor): The projected weigths.
    """
    p = centers.shape[0]
    d_out = target_evals.shape[-1]
    dtype = centers.dtype
    model_batch_size = min(10_000, preconditioner.critical_batch_size)

    projection_dataset = array_dataset.ArrayDataset(centers, target_evals)
    projection_loader = torch_data.DataLoader(
        projection_dataset, batch_size=model_batch_size, shuffle=True
    )

    # wiegths
    weights_project = torch.zeros(p, d_out, device=centers.device, dtype=dtype)

    for _ in range(1):
        for z_batch, labels_batch, id_batch in tqdm(
            projection_loader,
            total=len(projection_loader),
            leave=False,
            desc="Projection",
        ):
            kernel_mat = kernel_fn(z_batch, centers)
            batch_p = kernel_mat @ weights_project

            grad = batch_p - labels_batch.to(dtype).to(batch_p.device)
            batch_size = z_batch.shape[0]

            lr = preconditioner.scaled_learning_rate(batch_size)
            _, delta = preconditioner.delta(z_batch.to(grad.device).to(dtype), grad)
            weights_project[id_batch, :] = weights_project[id_batch, :] - lr * grad
            weights_project[nys_centers, :] = (
                weights_project[nys_centers, :] + lr * delta
            )

    return weights_project
