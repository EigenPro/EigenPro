from typing import Callable, List, Optional, Union
from eigenpro.utils.tensor import DistributedTensor, SingleDeviceTensor, BaseDeviceTensor
from eigenpro.utils.types import assert_and_raise
from eigenpro.utils.ops import (
        distributed_kernel_evaluation,
        distributed_matrix_multiply,
        distributed_matrix_slicing
    )
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
                 centers: torch.Tensor,
                 dtype: torch.dtype = torch.float32,
                 weights: torch.Tensor = None,
                 device_manager = None,
                ):
        self._kernel_fn = kernel_fn
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._size = size
        self._train = False
        self.device_manager = device_manager
        self.is_multi_device = (len(self.device_manager.devices) > 1)
        self.dtype = dtype
        
        self.init_weights(weights)
        self.init_centers(centers)

        if self.is_multi_device:
            chunk_sizes = device_manager.chunk_sizes(size)
            self._device_offsets = torch.cumsum(torch.as_tensor(chunk_sizes), 0, dtype=torch.int64) - chunk_sizes[0]
        else:
            self._device_offsets = torch.zeros(1,dtype=torch.int64)

    @property
    def device_offsets(self):
        return self._device_offsets

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
    def weights(self, weights=None):
        if isinstance(weights, DistributedTensor) or isinstance(weights, SingleDeviceTensor):
            self._weights = weights
        else:
            raise TypeError("input `weights` must be of type `DistributedTensor` or `SingleDeviceTensor`")

    @weights.deleter
    def weights(self):
        del self._weights
    
    def init_weights(self, weights):
        if weights is None:
            if self.is_multi_device:
                self.weights = DistributedTensor([
                        torch.zeros(c, self.n_outputs, dtype=self.dtype, device=d
                        ) for c, d in zip(self.device_manager.chunk_sizes(self.size), self.device_manager.devices)
                    ])
            else:
                self.weights = SingleDeviceTensor(
                    torch.zeros(
                        self.size, self.n_outputs, 
                        dtype=self.dtype, 
                        device=self.device_manager.base_device))
        elif isinstance(weights, torch.Tensor):
            self.weights = self.device_manager.scatter(weights)
        elif isinstance(weights, DistributedTensor):
            self.weights = weights
        else:
            raise ValueError

    def init_centers(self, centers):
        assert_and_raise(centers, torch.Tensor)
        self._centers = self.device_manager.scatter(centers)

    @property
    def centers(self) -> DistributedTensor:
        """Return the centers."""
        return self._centers # of type `DistributedTensor`


    def forward(self, 
            x: Union[torch.Tensor, SingleDeviceTensor], 
            cache_columns_by_idx: Union[torch.Tensor, SingleDeviceTensor] = None):
        """To add compatibility with other PyTorch models"""
        if self.is_multi_device:
            return self.forward_distributed(x, cache_columns_by_idx)
        # assert_and_raise(x, SingleDeviceTensor)
        assert_and_raise(self.centers, SingleDeviceTensor)
        assert_and_raise(self.weights, SingleDeviceTensor)
        kmat = self.kernel_fn(x, self.centers)
        preds = kmat @ self.weights
        if self._train:
            self._kmat_batch_centers_cached = kmat if cache_columns_by_idx is None else kmat[:, cache_columns_by_idx]
            del kmat
        return preds

    def forward_distributed(self, x, cache_columns_by_idx: DistributedTensor = None):
        assert_and_raise(x, DistributedTensor)
        assert_and_raise(self.centers, DistributedTensor)
        # if cache_columns_by_idx is not None:
        #     assert_and_raise(cache_columns_by_idx, DistributedTensor)
        assert x.num_parts==self.centers.num_parts
        kmat = distributed_kernel_evaluation(self.kernel_fn, x, self.centers)
        preds = distributed_matrix_multiply(kmat, self.weights)
        if self._train:
            if cache_columns_by_idx is None:
                self._kmat_batch_centers_cached = kmat 
            elif isinstance(cache_columns_by_idx, DistributedTensor):
                self._kmat_batch_centers_cached = distributed_matrix_slicing(kmat, cache_columns_by_idx)
            elif isinstance(cache_columns_by_idx, BaseDeviceTensor):
                self._kmat_batch_centers_cached = kmat.parts[self.device_manager.base_device_idx][:, cache_columns_by_idx]
        del kmat
        return preds

    def backward(self, grad):
        if self.is_multi_device:
            return self.backward_distributed(grad)
        assert_and_raise(grad, SingleDeviceTensor)
        if self._train:
            try:
                kg = self._kmat_batch_centers_cached.T @ grad
            except NameError:
                raise ValueError("must run `forward` once before calling `backward`.")
            del self._kmat_batch_centers_cached, grad
            return kg
        else:
            raise ValueError("method `KernelMachine.backward` cannot be invoked when model is not trainable. "
                "Try again after model.train()")

    def backward_distributed(self, grad):
        assert_and_raise(grad, DistributedTensor)
        if self._train:
            try:
                kg = distributed_matrix_multiply(self._kmat_batch_centers_cached.T, grad)
            except NameError:
                raise ValueError("must run `forward` once before calling `backward`.")
            del self._kmat_batch_centers_cached, grad
            return kg
        else:
            raise ValueError("method `KernelMachine.backward` cannot be invoked when model is not trainable. "
                "Try again after model.train()")
    
    # def update_weights_by_index(self, other_tensor, indices, alpha=1.0):
    #     if self.is_multi_device:
    #         assert_and_raise(other_tensor, DistributedTensor)
    #         assert_and_raise(indices, DistributedIndices)
    #         for dev_id, (w, idx) in enumerate(zip(self.weights.parts, indices)):
    #             w[dev_id][idx].add(other_tensor[dev_id], alpha=alpha)
    #     else:
    #         assert_and_raise(other_tensor, SingleDeviceTensor)
    #         assert_and_raise(indices, SingleDeviceTensor)
    #         self.weights[indices] += other_tensor

    # def update_weights(self, other_tensor, alpha=1.0):
    #     assert_and_raise(other_tensor, DistributedTensor)
    #     for dev_id, (w) in enumerate(self._weights):
    #         w[dev_id].add(other_tensor[dev_id], alpha=alpha)