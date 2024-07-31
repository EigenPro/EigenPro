import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as torch_data
import eigenpro.preconditioner as pcd
import eigenpro.data.array_dataset as array_dataset




def project(centers, target_evaluations, model_preconditioner, nys_model_indices, kernel_fn):

    """Project the target_evaluations into the function space span by centers given the kernel

    # ATTENTION: WORKS ONLY WITH 1-GPU!

     Args:
         centers (Tensor): Centers corresponding to some kernel function
         target_evaluations(Tensro): Evaluation of target function on the centers of the projection space.
         model_preconditioner: Preconditioner instance that contains a top kernel eigensystem for correcting
         the gradient forthe projection.
         nys_model_indices: indices of centers that used for preconditioner.
         kernel(function): kernel function.


     Returns:
         weights_project(Tensor): The projected weigths.
     """

    p = centers.shape[0]
    dtype = centers.dtype



    model_batch_size = min(10_000, model_preconditioner.critical_batch_size)



    # ipdb.set_trace()
    projection_dataset = array_dataset.ArrayDataset(centers, target_evaluations)
    projection_loader = torch_data.DataLoader(
        projection_dataset, batch_size=model_batch_size, shuffle=True)

    d_out = target_evaluations.shape[-1]
    #wiegths
    weights_project = torch.zeros(p, d_out,
                                  device=centers.device, dtype = dtype)

    # ipdb.set_trace()
    for _ in range(1):
        for z_batch, labels_batch, id_batch in tqdm(
                projection_loader,
                total=len(projection_loader),
                leave=False,
                desc="Projection"
        ):
            # optimizer.step(z_batch, grad_batch, id_batch, projection=True)

            kernel_mat = kernel_fn(z_batch, centers)
            batch_p = kernel_mat@weights_project

            grad = batch_p - labels_batch.to(dtype).to(batch_p.device)
            batch_size = z_batch.shape[0]

            lr = model_preconditioner.scaled_learning_rate(batch_size)
            deltap, delta = model_preconditioner.delta(
                z_batch.to(grad.device).to(dtype), grad)

            weights_project[id_batch,:] =  weights_project[id_batch,:]-lr*grad


            weights_project[nys_model_indices,:] = weights_project[nys_model_indices,:] + lr * delta


    return weights_project

