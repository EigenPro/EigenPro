"""Optimizer class and utility functions for training kernel machines.
"""

import models
import torch


def split_ids(ids: torch.Tensor, split_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits a tensor of ids into two based on a split id.

    Args:
        ids (torch.Tensor): A tensor of ids. split_id (int): The id to split the
        ids tensor on.

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
        elements 
            selected by batch_ids from the corresponding input tensor.
    """
    if not tensors:
        raise ValueError("At least one tensor should be provided.")
    
    if len(batch_ids) == len(tensors[0]):
        return tensors

    return tuple(tensor[batch_ids] for tensor in tensors)


class EigenPro(object):
    """EigenPro optimizer for kernel machines.

    Args:
        model: A KernelMachine instance.
        pcenters: Sampled centers for constructing EigenPro
            preconditioner. It is of shape (n_centers, n_features).

    Attributes:
        _model: A KernelMachine instance. _centers: Centers for preconditioner.
    """

    def __init__(self, model: models.KernelMachine, pcenters: torch.Tensor, threshold_index: int) -> None:
        self._model = self.shallow_copy(model)
        self._pcenters = pcenters
        self._pweights = torch.zeros((pcenters.shape[0], model.n_outputs))
        self._threshold_index = threshold_index

    def step(
        self, batch_x: torch.Tensor, batch_y: torch.Tensor, batch_ids: torch.Tensor
    ) -> None:
        """
                                            batch_x_existing    batch_x_new
        EP2         thd = 0     thd = 0     Y                   N EP3(paper)
        thd = p     thd = p     N                   Y EP3(genric) thd = 0
        thd = 0     Y                   Y
        """

        center_ids, data_ids = split_ids(batch_ids, self._threshold_index)

        # Computes delta for centers currently in the model and new centers.
        center_batch_x, center_batch_y = obtain_by_ids(center_ids, batch_x, batch_y)
        data_batch_x, data_batch_y = obtain_by_ids(data_ids, batch_x, batch_y)

        # Computes delta for centers currently in the model and new centers.
        center_delta = self.center_delta(center_batch_x, center_batch_y)
        data_delta = self.center_delta(data_batch_x, data_batch_y)

        # Computes delta for preconditioner centers.
        pdelta = self.pcenter_delta(batch_x, batch_y)

        # Obtains center weights to be updated.
        weights = obtain_by_ids(center_ids, self.model.weights)
        block_ids = self.model.add_centers(data_batch_x)
        extra_weights = self.model.get_weights(block_ids)

        # Updates weights of the model centers.
        self.center_update(center_delta, weights)
        self.center_update(data_delta, extra_weights)

        # Updates weights of the preconditioner centers.
        self.pcenter_update(pdelta, self._pweights)

    def pcenter_delta(
        self, batch_x: torch.Tensor, batch_y: torch.Tensor
    ) -> torch.Tensor:
        """Computes the delta of the weights for preconditioner centers."""
        pass

    def center_delta(
        self, batch_x: torch.Tensor, batch_y: torch.Tensor
    ) -> torch.Tensor:
        pass

    def pcenter_update(self, delta: torch.Tensor, weights: torch.Tensor) -> None:
        pass

    def center_update(self, delta: torch.Tensor, weights: torch.Tensor) -> None:
        pass

    def project(self) -> models.KernelMachine:
        pass
