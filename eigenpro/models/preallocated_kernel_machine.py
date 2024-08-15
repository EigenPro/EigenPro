from typing import Callable, Optional
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
        n_outputs: int,
        centers: torch.Tensor,
        dtype=torch.float32,
        tmp_centers_coeff: Optional[int] = 2,
        preallocation_size: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = "cpu",
    ) -> None:
        """Initializes the PreallocatedKernelMachine.

        Args:
            kernel_fn (Callable): The kernel function.
            n_outputs (int): The number of outputs.
            centers (torch.Tensor): The tensor of kernel centers.
            dtype: Data type, e.g. torch.float32 or torch.float16
            tmp_centers_coeff: the ratio between total centers(temporary included)
              and original centers
            preallocation_size (int, optional): The size for preallocation.
            weights (torch.Tensor, optional): The tensor of weights.
            device (torch.device, optional): The device for tensor operations.
        """
        super().__init__(kernel_fn, n_outputs, centers.shape[0])

        self.dtype = dtype
        self.tmp_centers_coeff = tmp_centers_coeff
        self.device = device
        self.original_size = centers.shape[0]
        self.nystrom_size = 0

        if preallocation_size is None:
            self._centers = torch.zeros(
                tmp_centers_coeff * centers.shape[0],
                centers.shape[1],
                device=device,
                dtype=self.dtype,
            )
            self._weights = torch.zeros(
                tmp_centers_coeff * centers.shape[0],
                self._n_outputs,
                device=device,
                dtype=self.dtype,
            )
        else:
            self._centers = torch.zeros(
                preallocation_size, centers.shape[1], device=device, dtype=self.dtype
            )
            self._weights = torch.zeros(
                preallocation_size, self._n_outputs, device=device, dtype=self.dtype
            )

        self.used_capacity = 0
        self.add_centers(centers, weights)

        self.lru = cache.LRUCache()

    @property
    def n_centers(self) -> int:
        """Return the number of centers."""
        return self.used_capacity

    @property
    def centers(self) -> int:
        """Return the centers."""
        return self._centers[: self.size]

    @property
    def weights(self) -> int:
        """Return the weights."""
        return self._weights[: self.size]

    def init_nystorm(self, nystrom_centers):
        self.centers_nyst = nystrom_centers.to(self.device)
        self.weights_nyst = torch.zeros((nystrom_centers.shape[0], self._n_outputs)).to(
            self.device
        )

    def forward(self, x: torch.Tensor, train=True) -> torch.Tensor:
        """Forward pass for the kernel machine.

        Args:
            x (torch.Tensor): input tensor of shape [n_samples, n_features].
            train(bool): Train mode, storing kernel_mat[:, :self.original_size].T
              in cache

        Returns:
            torch.Tensor: tensor of shape [n_samples, n_outputs].
        """

        x = x.to(self.dtype)
        x = x.to(self.device)

        centers = self._centers[: self.used_capacity, :]
        weights = self._weights[: self.used_capacity, :]

        kernel_mat = self._kernel_fn(x, centers[: self.original_size])

        p_orig = kernel_mat[:, : self.original_size] @ weights[: self.original_size, :]

        kernel_mat_nyst = self._kernel_fn(x, self.centers_nyst)
        p_nyst = kernel_mat_nyst @ self.weights_nyst

        if centers.shape[0] > self.original_size:
            p_tmp = fmm.KmV(
                self._kernel_fn,
                x,
                centers[self.original_size :],
                weights[self.original_size :],
                col_chunk_size=2**16,
            )
        else:
            p_tmp = 0

        predictions = p_orig + p_tmp + p_nyst

        if train:
            self.lru.put("k_centers_batch", kernel_mat[:, : self.original_size].T)

        del x, kernel_mat
        torch.cuda.empty_cache()
        return predictions

    def add_centers(
        self,
        centers: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        nystrom_centers: bool = False,
    ) -> torch.Tensor:
        """Adds centers and weights to the kernel machine.

        Args:
            centers (torch.Tensor): Tensor of shape [n_samples, n_features].
            weights (torch.Tensor, optional): Tensor of shape
              [n_samples, n_outputs].
            nystrom_centers (bool): True/False.
        Returns:
            None
        """

        if weights is not None:
            weights = weights.to(self.device)
        else:
            weights = torch.zeros(
                (centers.shape[0], self.n_outputs), device=self.device
            )

        if (
            self.used_capacity + centers.shape[0]
            > self.tmp_centers_coeff * self.original_size
        ):
            print("error")
            raise ValueError(
                f"Out of capacity for new centers: "
                f"{self.used_capacity=} > "
                f"{self.tmp_centers_coeff} * ({self.original_size=})"
            )

        self._centers[self.used_capacity : self.used_capacity + centers.shape[0], :] = (
            centers.to(self.dtype).to(self.device)
        )
        self._weights[self.used_capacity : self.used_capacity + centers.shape[0], :] = (
            weights
        )
        self.used_capacity += centers.shape[0]

        if nystrom_centers:
            self.nystrom_size = centers.shape[0]

        del centers, weights
        torch.cuda.empty_cache()

    def reset(self):
        """reset the model to before temporary centers were added.

        Args:
            No arguments.
        Returns:
            None
        """
        self.used_capacity = self.original_size
        self._centers[self.original_size :, :] = 0
        self._weights[self.original_size :, :] = 0
        self.weights_nyst = self.weights_nyst * 0

    def update_by_index(self, indices: torch.Tensor, delta: torch.Tensor) -> None:
        """Update the model weights by index.

        Args:
          indices: Tensor of 1-D indices to select rows of weights.
          delta: Tensor of weight update of shape [n_indices, n_outputs].
        """

        self._weights[indices] += delta

    def update_nystroms(self, update_wieghts):
        self.weights_nyst = self.weights_nyst + update_wieghts
