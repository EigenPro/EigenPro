from abc import abstractmethod
from typing import Callable, Optional

import torch


class KernelMachine:
    """Base class for KernelMachine.

    Attributes:
      kernel_fn: A callable function that computes the kernel matrix.
      n_outputs: The number of outputs.
    """

    def __init__(
        self,
        kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        n_outputs: int,
        size: int,
    ):
        self._kernel_fn = kernel_fn
        self._n_outputs = n_outputs
        self._size = size

    @property
    def kernel_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return self._kernel_fn

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def size(self) -> int:
        return self._size

    @property
    @abstractmethod
    def weights(self) -> int:
        """Return the weights."""
        raise NotImplementedError("Implement this in a subclass")

    @property
    @abstractmethod
    def centers(self) -> torch.Tensor:
        """Return the centers."""
        raise NotImplementedError("Implement this in a subclass")

    def __call__(self, *inputs):
        """To add compatibility with other PyTorch models"""
        return self.forward(*inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the kernel machine.

        Args:
            x (torch.Tensor): input tensor of shape [n_samples, n_features].

        Returns:
            torch.Tensor: tensor of shape [n_samples, n_outputs].
        """
        raise NotImplementedError

    def add_centers(
        self, centers: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Adds centers and weights to the kernel machine.

        Args:
            centers (torch.Tensor): Tensor of shape [n_samples, n_features].
            weights (torch.Tensor, optional): Tensor of shape
              [n_samples, n_outputs].

        Returns:
            torch.Tensor: Weight tensor of shape [n_samples, n_outputs].
        """
        raise NotImplementedError

    def update_by_index(self, indices: torch.Tensor, delta: torch.Tensor) -> None:
        """Update the model weights by index.

        Args:
          indices: Tensor of 1-D indices to select rows of weights.
          delta: Tensor of weight update of shape [n_indices, n_outputs].
        """
        raise NotImplementedError

    def shallow_copy(self) -> "KernelMachine":
        """A dummy shallow copy that returns the current instance."""
        return self
