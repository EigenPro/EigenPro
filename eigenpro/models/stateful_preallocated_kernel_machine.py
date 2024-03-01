from typing import Callable, List, Optional
import torch

import eigenpro.models.kernel_machine as km
import eigenpro.utils.cache as cache
import eigenpro.utils.fmm as fmm


class PreallocatedKernelMachine(km.KernelMachine):
    """Kernel machine class for handling kernel methods.

    Attributes:
        n_outputs: The number of outputs.
        n_centers: The number of model centers.
        _centers: A tensor of kernel centers of shape [n_centers, n_features].
        _weights: An optional tensor of weights of shape [n_centers, n_outputs].
    """

    def __init__(
            self,
            kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            n_inputs: int,
            n_outputs: int,
            size: int,
            dtype=torch.float32,
            weights: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = 'cpu'
        ) -> None:
        """Initializes the PreallocatedKernelMachine.
        """
        super().__init__(kernel_fn, n_inputs, n_outputs, size)

        self.dtype = dtype
        self.device = device

        self.reset()


    @property
    def n_centers(self) -> int:
        """Return the number of centers."""
        return self.used_capacity


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the kernel machine.

        Args:
                x (torch.Tensor): input tensor of shape [n_samples, n_features].
                projection(bool): Projection mode, updating projection weights
                train(bool): Train mode, storing kernel_mat[:, :self.original_size].T
                    in cache

        Returns:
                torch.Tensor: tensor of shape [n_samples, n_outputs].
        """
        return fmm.KmV(
                self.kernel_fn, x, 
                self.centers[:self.used_capacity],
                self.weights[:self.used_capacity],
                col_chunk_size=2**16
        )


    def add_centers(self,
            new_centers: torch.Tensor,
            new_weights: torch.Tensor,
        ) -> torch.Tensor:
        """Adds centers and weights to the kernel machine.
        """

        if (self.used_capacity + len(new_centers) > self.size):
            print("error")
            raise ValueError(f"Running out of capacity for new centers: ")

        self._centers[self.used_capacity: self.used_capacity + len(new_centers),:] = new_centers
        self._weights[self.used_capacity: self.used_capacity + len(new_weights),:] = new_weights
        self.used_capacity += len(new_centers)

        del new_centers, new_weights
        torch.cuda.empty_cache()


    def reset(self):
        """reset the model to before temporary centers were added.

        Args:
                No arguments.
        Returns:
                None
        """
        self.used_capacity = 0
        self.centers = torch.zeros(self.size, self.n_inputs, device=self.device, dtype=self.dtype)
        self.weights = torch.zeros(self.size, self.n_outputs, device=self.device, dtype=self.dtype)
