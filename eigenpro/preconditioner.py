"""
    Preconditioner class and utility functions for EigenPro iteration.
"""
from typing import Callable
import numpy as np
import torch
from eigenpro.utils import svd
from eigenpro.utils.cache import LRUCache

import ipdb


class KernelEigenSystem(svd.EigenSystem):
    """Extends the EigenSystem class for EigenPro kernel preconditioning.

    Attributes:
        _beta (float): (Approximate) maxinum of kernel norm.

    Attributes from EigenSystem:
        min_value (float): Smallest eigenvalue of the eigensystem.
        size (int): Size of the eigensystem.
        values (np.ndarray): The eigenvalues in descending order.
        vectors (np.ndarray): The eigenvectors in the same order of values.
    """

    def __init__(self, eigensys: svd.EigenSystem, beta: float) -> None:
        """Initialize an KernelEigenSystem instance.

        Args:
            eigensys (svd.EigenSystem): An instance of EigenSystem to extend.
            beta (float): (Approximate) maxinum of kernel norm.
        """
        # Creates an Adapter instance (KernelEigenSystem) from an EigenSystem
        # instance. This Adapter instance has all attributes and methods of the
        # given EigenSystem instance.
        self.__dict__ = eigensys.__dict__
        self._beta = beta
        # Overwrites base class `_vectors`
        self._vectors = torch.as_tensor(eigensys.vectors, dtype=torch.float32)
        self._normalized_ratios = torch.Tensor(
            (1 - eigensys.min_value / eigensys.values) / eigensys.values)

    @property
    def beta(self) -> float:
        """Get the beta parameter.

        Returns:
            float: The beta parameter.
        """
        return self._beta

    @property
    def normalized_ratios(self) -> np.ndarray:
        """Calculates the normalized eigenvalue ratio for kernel operator.

        Returns:
            np.ndarray: The computed ratios.
        """
        return self._normalized_ratios

    def change_type(self, dtype=torch.float32):
        """Convert to lower precision
        Args:
            type (torch.type): it is either torch.float32 or torch.float16
        Returns:
            None
        Raises:
            None: This method is not expected to raise any exceptions.
        """
        self._vectors = self._vectors.to(dtype)
        self._normalized_ratios =  self._normalized_ratios.to(dtype)


def top_eigensystem(samples: torch.Tensor, q: int,
                    kernel_fn: Callable[[torch.Tensor, torch.Tensor],
                                        torch.Tensor]
                    ) -> KernelEigenSystem:
    """Computes the top eigen system for a kernel matrix.

    Args:
        samples (torch.Tensor): A tensor containing the samples.
        q (int): The number of top eigenvalues to consider from the
            eigenspectrum.
        kernel_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): 
            A kernel function that computes the kernel matrix for two sets of
            samples.

    Returns:
        A KernelEigenSystem instance

    """
    n_sample = samples.shape[0]
    kernel_mat = kernel_fn(samples, samples).cpu().data.numpy()
    # Obtains eigensystem for the normalized kernel matrix.
    eigensys = svd.top_q_eig(kernel_mat / n_sample, q)
    
    # Obtains an upper bound for ||k(x, \cdot)||.
    beta = max(np.diag(kernel_mat))

    return KernelEigenSystem(eigensys, beta)


class Preconditioner:
    """Class for preconditioning based on a given kernel function and centers.

    Attributes:
        kernel_fn: Callable kernel function that takes centers and inputs,
            and returns a kernel matrix.
        centers: Tensor representing kernel centers of shape [n_centers,
            n_features].
        weights: Weight parameters of shape [n_centers, n_outputs] corresponding
            to the centers.
        top_q_eig: Construct top q eigensystem for preconditioning.
    """

    def __init__(self,
                 kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 centers: torch.Tensor,
                 weights: torch.Tensor,
                 top_q_eig: int) -> None:
        """Initializes the Preconditioner."""
        self._kernel_fn = kernel_fn
        self._centers = centers
        self._weights = weights
        self._eigensys = top_eigensystem(centers, top_q_eig, kernel_fn)

    @property
    def eigensys(self) -> torch.Tensor:
        """Returns eigensys of preconditioner."""
        return self._eigensys
    @property
    def centers(self) -> torch.Tensor:
        """Returns centers for constructing the preconditioner."""
        return self._centers

    @property
    def critical_batch_size(self) -> int:
        """Computes and returns the critical batch size."""
        return int(self._eigensys.beta / self._eigensys.min_value)
    
    @property
    def weights(self) -> torch.Tensor:
        """Returns weights corresponding to the centers."""
        return self._weights

    def learning_rate(self, batch_size: int) -> float:
        """Computes and returns the learning rate based on the batch size."""
        if batch_size < self.critical_batch_size:
            return batch_size / self._eigensys.beta / 2
        else:
            return batch_size / (self._eigensys.beta +
                                 (batch_size - 1) * self._eigensys.min_value)

    def scaled_learning_rate(self, batch_size: int) -> float:
        """Computes and returns the scaled learning rate."""
        return 2 / batch_size * self.learning_rate(batch_size)

    def delta(self, batch_x: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Computes weight delta for preconditioner centers.
        Args:
            batch_x (torch.Tensor): Of shape `[, n_features]`.
            grad (torch.Tensor): Of shape `[batch_size, n_outputs]`.

        Returns:
            torch.Tensor, torch.Tensor: Of Shape `[q, n_outputs]`, `[n_centers, n_outputs]` respectively.

        Raises:
            None: This method is not expected to raise any exceptions.
        """
        kernel_mat = self._kernel_fn(self._centers, batch_x)
        # kg is of shape [n_centers, n_outputs]
        kg = kernel_mat @ grad
        eigenvectors = self._eigensys.vectors
        normalized_ratios = self._eigensys.normalized_ratios

        # vtkg of shape [q, n_outputs]
        vtkg = eigenvectors.T @ kg
        vdvtkg = eigenvectors @ (normalized_ratios * vtkg)

        return vtkg,vdvtkg

    def eval_vec(self,batch):
        """Computes K(X_s,batch)@(D*E) which is a part of correction term
        Args:
            batch (torch.Tensor): Of shape `[, n_features]`.
        Returns:
            torch.Tensor: Of Shape `[batch_size, q]`

        Raises:
            None: This method is not expected to raise any exceptions.
        """
        eigenvectors = self._eigensys.vectors
        normalized_ratios = self._eigensys.normalized_ratios
        return self._kernel_fn(batch,self.centers)@ (normalized_ratios*eigenvectors)

    def change_type(self, dtype=torch.float32):
        """Converting to half precision
        Args:
            type (torch.type): it is either torch.float32 or torch.float16
        Returns:
            None
        Raises:
            None: This method is not expected to raise any exceptions.
        """
        self._eigensys.change_type(dtype)
        self._centers = self.centers.to(dtype)


