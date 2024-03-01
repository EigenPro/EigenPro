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
                 n_inputs: int,
                 n_outputs: int,
                 size: int,
                ):
        self._kernel_fn = kernel_fn
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._size = size
        self._train = False

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # remap
    def __call__(self, *args):
        return self.forward(*args)

    @property
    def kernel_fn(self) -> Callable[ [torch.Tensor, torch.Tensor], torch.Tensor ]:
        return self._kernel_fn
    
    @property
    def n_inputs(self) -> int:
        return self._n_inputs

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
        if self._train:
            self._kmat_batch_centers_cached = self.kernel_fn(x, self.centers)
            return self._kmat_batch_centers_cached @ self.weights
        else:
            return self.kernel_fn(x, self.centers) @ self.weights

    def backward(self, grad):
        if self._train:
            kg = self._kmat_batch_centers_cached.T @ grad
            del self._kmat_batch_centers_cached
            return kg
        else:
            raise ValueError("method `KernelMachine.backward` cannot be invoked when model is not trainable. "
                "Try again after model.train()")
    
    def update_weights_by_index(
            self, indices: torch.Tensor,
            delta: torch.Tensor
        ) -> None:
        """Update the model weights by index.
        
        Args:
            indices: Tensor of 1-D indices to select rows of weights.
            delta: Tensor of weight update of shape [n_indices, n_outputs].
        """
        self._weights[indices] += delta