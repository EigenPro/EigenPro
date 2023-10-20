"""Optimizer class and utility functions for EigenPro iteration."""
import torch

import models
import preconditioner as pcd


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
        model (models.KernelMachine): A KernelMachine instance.
        threshold_index (int): An index used for thresholding.
        precon (pcd.Preconditioner): Preconditioner instance that contains a
            top kernel eigensystem for correcting the gradient.

    Attributes:
        model (models.KernelMachine): A KernelMachine instance.
        precon (pcd.Preconditioner): A Preconditioner instance.
        _threshold_index (int): An index used for thresholding.
    """

    def __init__(self,
                 model: models.KernelMachine,
                 threshold_index: int,
                 precon: pcd.Preconditioner) -> None:
        """Initialize the EigenPro optimizer."""
        self._model = model.shallow_copy()
        self._threshold_index = threshold_index

        self._precon = precon
        self._model.add_centers(precon.centers, precon.weights)

        ###### Amirhesam: TODO: not sure cat is the best way?
        self.grad_accumulation = 0


        model.forward(self._precon.centers)
        precon_eigenvectors = precon.eigensys.vectors
        self.k_centers_nystroms_mult_eigenvecs =\
            model.lru.get('k_centers_batch_grad').to(precon_eigenvectors.device)@precon_eigenvectors


    @property
    def model(self) -> models.KernelMachine:
        """Gets the active model (for training).

        Returns:
            models.KernelMachine: The active model.
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
        pbatch = self.model(batch_x)
        k_centers_batch_grad = self.model.lru.get('k_centers_batch_grad')
        grad = pbatch - batch_y
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

        delta, deltap = self.precon.delta(batch_x, grad)
        self.grad_accumulation = self.grad_accumulation + k_centers_batch_grad - \
                                 self.k_centers_nystroms_mult_eigenvecs @ deltap

        if in_batch_size:
            self.model.update_by_index(in_ids, in_delta)
        if out_batch_size:
            self.model.add_centers(batch_x[out_ids], out_delta)

        self.precon.update(delta, len(batch_ids))
