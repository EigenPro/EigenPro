from typing import Callable, List, Optional
from eigenpro.utils.tensor import DistributedTensor, SingleDeviceTensor
from eigenpro.utils.types import assert_and_raise
import torch



def distributed_kernel_evaluation(kernel_fn, X: DistributedTensor, centers: DistributedTensor):
    with ThreadPoolExecutor() as executor:
        out = [
                executor.submit(kernel_fn, x, z) for x, z in zip(X, centers)
            ]
        kmat = DistributedTensor([k.result() for k in out])
        del out
    return kmat

def distributed_matrix_multiply(mat1: DistributedTensor, mat2: DistributedTensor):
    with ThreadPoolExecutor() as executor:
        out = [
                executor.submit(torch.matmul, m1, m2) for m1, m2 in zip(mat1, mat2)
            ]
        mat3 = DistributedTensor([k.result() for k in out])
        del out
    return mat3

def distributed_matrix_slicing(mat: DistributedTensor, idx: DistributedTensor, offsets: torch.Tensor):
    with ThreadPoolExecutor() as executor:
        out = [
                executor.submit(torch.index_select, m, 0, o+i) for m, i, o in zip(mat, idx, offsets)
            ]
        mat3 = DistributedTensor([k.result() for k in out])
        del out
    return mat3


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
            self._device_offsets = torch.cumsum(chunk_sizes, dtype=torch.int64) - chunk_sizes[0]
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

    def init_weights(self, weights):
        if weights is None:
            if self.is_multi_device:
                self.weights = DistributedTensor([
                        torch.zeros(c, self.n_outputs, dtype=self.dtype, device=d
                        ) for c, d in zip(self.device_manager.chunk_sizes(self.size), self.device_manager.devices)
                    ])
            else:
                self.weights = SingleDeviceTensor(torch.zeros(self.size, self.n_outputs, dtype=self.dtype, device=self.device_manager.base_device))
        elif isinstance(weights, torch.Tensor):
            self.weights = device_manager.scatter(weights)
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


    def forward(self, x: torch.Tensor, cache_columns_by_idx: torch.Tensor = None):
        """To add compatibility with other PyTorch models"""
        if self.is_multi_device:
            return self.forward_distributed(x, cache_columns_by_idx)
        assert_and_raise(x, SingleDeviceTensor)
        assert_and_raise(self.centers, SingleDeviceTensor)
        assert_and_raise(self.weights, SingleDeviceTensor)
        kmat = self.kernel_fn(x, self.centers)
        preds = kmat @ self.weights
        if self._train:
            self._kmat_batch_centers_cached = kmat if cache_columns_by_idx is None else kmat[:, cache_columns_by_idx]
            del kmat
        return preds

    def forward_distributed(self, x, cache_columns_by_idx: DistributedTensor):
        assert_and_raise(x, DistributedTensor)
        assert_and_raise(self.centers, DistributedTensor)
        assert_and_raise(cache_columns_by_idx, DistributedTensor)
        assert len(x)==len(self.centers)
        kmat = distributed_kernel_evaluation(self.kernel_fn, x, self.centers)
        preds = distributed_matrix_multiply(kmat, self.weights)
        if self._train:
            if cache_columns_by_idx is None:
                self._kmat_batch_centers_cached = kmat 
            else:
                self._kmat_batch_centers_cached = distributed_matrix_slicing(kmat, cache_columns_by_idx)
        del kmat
        return preds

    def backward(self, grad):
        if self.is_multi_device:
            return self.backward_distributed(self, grad)
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
    
    def update_weights_by_index(self, other_tensor, indices, alpha=1.0):
        if self.is_multi_device:
            assert_and_raise(other_tensor, DistributedTensor)
            assert_and_raise(indices, DistributedIndices)
            for dev_id, (w, idx) in enumerate(zip(self._weights, indices)):
                w[dev_id][idx].add(other_tensor[dev_id], alpha=alpha)
        else:
            assert_and_raise(other_tensor, SingleDeviceTensor)
            assert_and_raise(indices, SingleDeviceTensor)
            self.weights[indices] += other_tensor

    def update_weights(self, other_tensor, alpha=1.0):
        assert_and_raise(other_tensor, DistributedTensor)
        for dev_id, (w) in enumerate(self._weights):
            w[dev_id].add(other_tensor[dev_id], alpha=alpha)