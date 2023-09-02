"""
    Preconditioner class and utility functions for EigenPro iteration.
"""
from typing import Callable
import numpy as np
import torch
import svd


class KernelEigenSystem(svd.EigenSystem):
    """Extends the EigenSystem class for EigenPro kernel preconditioning.

    Attributes:
        _beta (float): (Approximate) maxinum of kernel norm.
        _scale (float): Scale factor.
    """

    def __init__(self, eigensys: svd.EigenSystem, beta: float, scale: float
                 ) -> None:
        """Initialize an KernelEigenSystem instance.

        Args:
            eigensys (svd.EigenSystem): An instance of EigenSystem to extend.
            beta (float): (Approximate) maxinum of kernel norm.
            scale (float): The scale factor.
        """
        self.__dict__ = eigensys.__dict__
        self._beta = beta
        # TODO(s1van): Move scale factor into EigenSystem by directly
        # normalizing the eigenvalues.
        self._scale = scale
        # Overwrites base class `_vectors`
        self._vectors = torch.as_tensor(eigensys.vectors, dtype=torch.float32)
        self._normalized_ratios = torch.Tensor(
            self._scale * (1 - eigensys.min_value / eigensys.values) / eigensys.values)

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
    eigensys = svd.top_q_eig(kernel_mat, q)
    
    # Obtains an upper bound for ||k(x, \cdot)||.
    beta = max(np.diag(kernel_mat))

    return KernelEigenSystem(eigensys, beta, 1 / n_sample)


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
            batch_x (torch.Tensor): Of shape `[batch_size, n_features]`.
            grad (torch.Tensor): Of shape `[batch_size, n_outputs]`.

        Returns:
            torch.Tensor: Of Shape `[n_centers, n_outputs]`.

        Raises:
            None: This method is not expected to raise any exceptions.
        """
        kernel_mat = self._kernel_fn(self._centers, batch_x)
        # of shape [n_centers, n_outputs]
        kg = kernel_mat @ grad
        eigenvectors = self._eigensys.vectors
        normalized_ratios = self._eigensys.normalized_ratios

        # of shape [q, n_outputs]
        vtkg = eigenvectors.T @ kg
        vdvtkg = eigenvectors @ (normalized_ratios * vtkg)

        return vdvtkg

    def update(self, delta: torch.Tensor, batch_size: int) -> None:
        """Updates the weight parameters."""
        lr = self.scaled_learning_rate(batch_size)
        self._weights.add_(lr * delta)
