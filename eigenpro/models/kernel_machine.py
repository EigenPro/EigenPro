from typing import Callable, List, Optional

import torch


class KernelMachine:
    """Base class for KernelMachine.
    
    Attributes:
        kernel_fn: A callable function that computes the kernel matrix.
        n_outputs: The number of outputs.
    """
    def __init__(self,
                 kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 n_outputs: int,
                 size: int
                ):
        self._kernel_fn = kernel_fn
        self._n_outputs = n_outputs
        self._size = size

    # remap
    def __call__(self, *args):
        return self.forward(*args)

    @property
    def kernel_fn(self) -> Callable[[torch.Tensor, torch.Tensor],
                                                                    torch.Tensor]:
        return self._kernel_fn
    
    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @property
    def size(self) -> int:
        return self._size

    @property
    def weights(self) -> int:
        """Return the weights."""
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def centers(self) -> torch.Tensor:
        """Return the centers."""
        return self._centers

    @centers.setter
    def centers(self, centers):
        self._centers = centers
    


    def forward(self, x):
        """To add compatibility with other PyTorch models"""
        self._kmat_batch_centers_cached = self.kernel_fn(x, self.centers)
        return self._kmat_batch_centers_cached @ self.weights

    def backward(self, grad):
        kg = self._kmat_batch_centers_cached.T @ grad
        del self._kmat_batch_centers_cached
        return kg



    def add_centers(self, 
        centers: torch.Tensor,
        weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adds centers and weights to the kernel machine.
        
        Args:
                centers (torch.Tensor): Tensor of shape [n_samples, n_features].
                weights (torch.Tensor, optional): Tensor of shape
                    [n_samples, n_outputs].
                
        Returns:
                torch.Tensor: Weight tensor of shape [n_samples, n_outputs].
        """
        self._centers = torch.cat([self._centers, centers])
        weights_ = torch.zeros(self.size, self.n_outputs) if weights is None else weights
        self.weights = torch.cat([self._weights, weights_])
    
    def update_by_index(self, indices: torch.Tensor,
                                            delta: torch.Tensor) -> None:
        """Update the model weights by index.
        
        Args:
            indices: Tensor of 1-D indices to select rows of weights.
            delta: Tensor of weight update of shape [n_indices, n_outputs].
        """
        raise NotImplementedError
    
    def shallow_copy(self) -> 'KernelMachine':
        """A dummy shallow copy that returns the current instance."""
        return self
