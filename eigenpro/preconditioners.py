"""
    Preconditioner class and utility functions for EigenPro iteration.
"""
from typing import Callable

import torch

import eigenpro.utils.cache as cache
import eigenpro.utils.keigh as keigh
import eigenpro.utils.fmm as fmm
import functools

class Preconditioner:
    """Class for preconditioning based on a given kernel function and centers.

    Attributes:
        kernel_fn: Callable kernel function that takes centers and inputs,
            and returns a kernel matrix.
        centers: Tensor representing kernel centers of shape [n_centers,
            n_features].
        top_q_eig: Construct top q eigensystem for preconditioning.
    """

    def __init__(self,
                 kernel_fn: Callable[[torch.Tensor, torch.Tensor],
                                     torch.Tensor],
                 centers: torch.Tensor,
                 top_q_eig: int,
                 center_ids: torch.Tensor = None,
                ) -> None:
        """Initializes the Preconditioner."""
        self._kernel_fn = kernel_fn
        self._eigensys = keigh.top_eigensystem(centers, top_q_eig, kernel_fn)
        self._centers = centers
        self._center_ids = center_ids
        self.size = centers.shape[0]
        self.lru = cache.LRUCache()
        self.normalized_eigenvectors = self._eigensys.vectors * self._eigensys.normalized_ratios.sqrt()

    # @property
    # def eigensys(self) -> torch.Tensor:
    #     """Returns eigensys of preconditioner."""
    #     return self._eigensys

    @property
    def center_ids(self) -> torch.Tensor:
        """Returns centers for constructing the preconditioner."""
        return self._center_ids

    @property
    def centers(self) -> torch.Tensor:
        """Returns centers for constructing the preconditioner."""
        return self._centers

    @property
    def critical_batch_size(self) -> int:
        """Computes and returns the critical batch size."""
        return int(self._eigensys.beta / self._eigensys.min_value)

    @functools.cache
    def learning_rate(self, batch_size: int) -> float:
        """Computes and returns the learning rate based on the batch size."""
        if batch_size < self.critical_batch_size:
            return batch_size / self._eigensys.beta/2
        else:
            return batch_size / (self._eigensys.beta +
                                 (batch_size - 1) * self._eigensys.min_value)

    @functools.cache
    def scaled_learning_rate(self, batch_size: int) -> float:
        """Computes and returns the scaled learning rate."""
        return float(2 / batch_size * self.learning_rate(batch_size))


    # def delta(
    #     self, 
    #     batch_x: torch.Tensor, 
    #     grad: torch.Tensor
    # ) -> torch.Tensor:
    #     """Computes weight delta for preconditioner centers.
    #     Args:
    #         batch_x (torch.Tensor): Of shape `[, n_features]`.
    #         grad (torch.Tensor): Of shape `[batch_size, n_outputs]`.

    #     Returns:
    #         torch.Tensor: Of Shape `[q, n_outputs]`
    #         torch.Tensor: Of Shape `[n_centers, n_outputs]`

    #     Raises:
    #         None: This method is not expected to raise any exceptions.
    #     """
    #     device = batch_x.device

    #     kernel_mat = self._kernel_fn(self._centers.to(device), batch_x)
    #     kg = kernel_mat @ grad
    #     eigenvectors = self._eigensys.vectors.to(device)
    #     normalized_ratios = self._eigensys.normalized_ratios.to(device)

    #     vtkg = eigenvectors.T @ kg
    #     vdvtkg = (normalized_ratios*eigenvectors ) @ vtkg

    #     del eigenvectors, normalized_ratios

    #     return vtkg, vdvtkg


    # def change_type(self, dtype=torch.float32):
    #     """Converting to half precision
    #     Args:
    #         type (torch.type): it is either torch.float32 or torch.float16
    #     Returns:
    #         None
    #     Raises:
    #         None: This method is not expected to raise any exceptions.
    #     """
    #     self._eigensys.change_type(dtype)
    #     self._centers = self.centers.to(dtype)
