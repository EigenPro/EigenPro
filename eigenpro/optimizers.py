"""Optimizer class and utility functions for EigenPro iteration."""
import torch
from .models import KernelMachine
from .preconditioner import Preconditioner

def split_ids(ids: torch.Tensor, split_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits a tensor of ids into two based on a split id.

    Args:
        ids (torch.Tensor): A tensor of ids.
        split_id (int): The id to split the ids tensor on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two tensors, the first contains ids
            less than or equal to split_id, the second contains ids greater than
            split_id.
    """
    leq_ids = ids[ids <= split_id]
    g_ids = ids[ids > split_id]
    return leq_ids, g_ids


def obtain_by_ids(batch_ids: torch.Tensor, *tensors: torch.Tensor
                  ) -> tuple[torch.Tensor, ...]:
    """Obtain elements in tensors by indices specified in batch_ids.

    Args:
        batch_ids (torch.Tensor): A tensor of indices to select from the
        tensors. *tensors (torch.Tensor): A variable number of tensors to select
        data from using batch_ids.

    Returns:
        Tuple[torch.Tensor, ...]: A tuple of tensors, each tensor contains
        elements selected by batch_ids from the corresponding input tensor.
    """
    if not tensors:
        raise ValueError("At least one tensor should be provided.")
    
    if len(batch_ids) == 0:
        ret = tuple(torch.tensor([], dtype=tensor.dtype) for tensor in tensors)
    else:
    # Handle empty tensors separately to avoid "index out of bounds" error
        ret = tuple(
            tensor[batch_ids] if len(tensor) > 0 else torch.tensor([], dtype=tensor.dtype) 
            for tensor in tensors
        )
    
    if len(tensors) == 1:
        return ret[0]
    
    return ret


class EigenPro:
    """EigenPro optimizer for kernel machines.

    Args:
        model (KernelMachine): A KernelMachine instance.
        threshold_index (int): An index used for thresholding.
        precon_data (Preconditioner): Preconditioner instance that contains a
            top kernel eigensystem for correcting the gradient for data.
        precon_model (Preconditioner): Preconditioner instance that contains a
            top kernel eigensystem for correcting the gradient for the projection

    Attributes:
        model (KernelMachine): A KernelMachine instance.
        precon (Preconditioner): A Preconditioner instance.
        _threshold_index (int): An index used for thresholding.
    """

    def __init__(self,
                 model: KernelMachine,
                 threshold_index: int,
                 precon_data: Preconditioner,
                 precon_model: Preconditioner,
                 kz_xs_evecs:torch.tensor = None,
                 type=torch.float32,
                 accumulated_gradients:bool = False,) -> None:
        """Initialize the EigenPro optimizer."""
        self._model = model.shallow_copy()
        self._threshold_index = threshold_index

        self._precon = precon
        self._model.add_centers(precon.centers, precon.weights)
        self.grad_accumulation = 0


        model.forward(self._precon.centers)
        precon_eigenvectors = precon.eigensys.vectors
        self.k_centers_nystroms_mult_eigenvecs =\
            model.lru.get('k_centers_batch_grad').to(precon_eigenvectors.device) @ precon_eigenvectors


    @property
    def model(self) -> KernelMachine:
        """Gets the active model (for training).

        Returns:
            KernelMachine: The active model.
        """
        return self._model

    @property
    def precon(self) -> pcd.Preconditioner:
        """Gets the preconditioner.

        Returns:
            pcd.Preconditioner: The preconditioner.
        """
        return self._precon

    def step(self,
             batch_x: torch.Tensor,
             batch_y: torch.Tensor,
             batch_ids: torch.Tensor) -> None:
        """Performs a single optimization step.

        Args:
            batch_x (torch.Tensor): Batch of input features.
            batch_y (torch.Tensor): Batch of target values.
            batch_ids (torch.Tensor): Batch of sample indices.
        """
        in_ids, out_ids = split_ids(batch_ids, self._threshold_index)
        batch_p = self.model(batch_x)
        k_centers_batch_grad = self.model.lru.get('k_centers_batch_grad')
        grad = batch_p - batch_y
        in_batch_g = obtain_by_ids(in_ids, grad)
        out_batch_g = obtain_by_ids(out_ids, grad)

        in_batch_size = len(in_batch_g)
        if in_batch_size:
            in_delta = -self.precon.scaled_learning_rate(
                in_batch_size) * in_batch_g

        out_batch_size = len(out_batch_g)
        if out_batch_size:
            out_delta = -self.precon.scaled_learning_rate(
                out_batch_size) * out_batch_g

        deltap, delta = self.precon.delta(batch_x, grad)
        self.grad_accumulation = self.grad_accumulation + k_centers_batch_grad - \
                                 self.k_centers_nystroms_mult_eigenvecs @ deltap

        if self.grad_accumulation is None or projection:
            self.model.update_by_index(batch_ids, -lr *grad,projection=projection )
        else:
            k_centers_batch_all = self.model.lru.get('k_centers_batch')
            self.model.lru.cache.clear()
            kgrads = []
            for k in k_centers_batch_all:
                kgrads.append((k @ grad.to(k.device)))
            k_centers_batch_grad = torch.cat(kgrads)  ##  K(bathc,Z) (f-y)

            self.grad_accumulation = self.grad_accumulation - lr*\
                                     ( k_centers_batch_grad -
                                       (self.k_centers_nystroms_mult_eigenvecs @
                                        deltap) )

            self.model.add_centers(batch_x, -lr*grad)
            # print(f'used capacity:{self.model.shard_kms[0].used_capacity}')

            del k_centers_batch_grad, kgrads, k_centers_batch_all,batch_y




        self.model.update_by_index(torch.tensor(list(range(self.precon_model._centers.shape[0])))
                                   ,lr*delta, nystrom_update=True,projection=projection)

        del grad, batch_x, batch_p, deltap, delta
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def reset(self):
        """Reset the gradient accumulation
        Args:
            None
        return:
            None
        """
        self.grad_accumulation = 0

        self.precon.update(delta, len(batch_ids))
