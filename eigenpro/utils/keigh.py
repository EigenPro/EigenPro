"""`eigh` for kernel matrices"""

from typing import Callable

import numpy as np
import torch

import eigenpro.utils.eigh as eigh


class KernelEigenSystem(eigh.EigenSystem):
    """Extends the EigenSystem class for EigenPro kernel preconditioning.

    Attributes:
        _beta (float): (Approximate) maxinum of kernel norm.

    Attributes from EigenSystem:
        min_value (float): Smallest eigenvalue of the eigensystem.
        size (int): Size of the eigensystem.
        values (np.ndarray): The eigenvalues in descending order.
        vectors (np.ndarray): The eigenvectors in the same order of values.
    """

    def __init__(self, eigensys: eigh.EigenSystem, beta: float) -> None:
        """Initialize an KernelEigenSystem instance.

        Args:
            eigensys (EigenSystem): An instance of EigenSystem to extend.
            beta (float): (Approximate) maxinum of kernel norm.
        """
        # Creates an Adapter instance (KernelEigenSystem) from an EigenSystem
        # instance. This Adapter instance has all attributes and methods of the
        # given EigenSystem instance.
        self.__dict__ = eigensys.__dict__
        self._beta = beta
        # Overwrites base class `_vectors`
        self._vectors = torch.as_tensor(eigensys.vectors)

        self._normalized_ratios = (
            torch.as_tensor(
                (1 - eigensys.min_value / eigensys.values) / eigensys.values
            )
            / eigensys.vectors.shape[0]
        )

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
        self._normalized_ratios = self._normalized_ratios.to(dtype)


def top_eigensystem(
    samples: torch.Tensor,
    q: int,
    kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
    kernel_mat = kernel_fn(samples, samples)
    # Obtains eigensystem for the normalized kernel matrix.
    eigensys = eigh.top_q_eig(kernel_mat / n_sample, q)

    # Obtains an upper bound for ||k(x, \cdot)||.
    beta = max(torch.diag(kernel_mat))
    return KernelEigenSystem(eigensys, beta)
