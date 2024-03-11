"""Optimizer class and utility functions for EigenPro iteration."""
import torch
from concurrent.futures import ThreadPoolExecutor
import eigenpro.kernel_machine as km
import eigenpro.preconditioner as pcd
from eigenpro.utils.tensor import DistributedTensor, SingleDeviceTensor, RowDistributedTensor
from eigenpro.utils.ops import distributed_kernel_evaluation
from eigenpro.utils.types import assert_and_raise


from typing import Callable, List, Optional, Union
import torch

import eigenpro.utils.fmm as fmm


def slice_distributed_matrix_till_capacity(mat: RowDistributedTensor, capacities: torch.Tensor):
    with ThreadPoolExecutor() as executor:
        out = [
                executor.submit(torch.index_select, m, 0, torch.arange(c, dtype=torch.int64, device=m.device)) for m, c in zip(mat.parts, capacities)
            ]
        mat3 = RowDistributedTensor([k.result() for k in out], base_device_idx=mat.base_device_idx)
        del out
    return mat3

class LatentKernelMachine(km.KernelMachine):
    """Kernel machine class for handling kernel methods.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a KernelMachine with fixed size.
        """
        self.init_centers = self.reset_centers
        self.init_weights = self.reset_weights

        super().__init__(*args, **kwargs)

        self._used_capacity = None
        self.reset()

    def reset_centers(self, centers=None):
        if self.is_multi_device:
            self._centers = RowDistributedTensor.zeros(
                self.device_manager.chunk_sizes(self.size), 
                self.n_inputs, 
                self.dtype, 
                self.device_manager.devices,
                self.device_manager.base_device_idx
            )
        else:
            self._centers = SingleDeviceTensor(torch.zeros(self.size, self.n_inputs, dtype=self.dtype, device=self.device_manager.base_device))

    def reset_weights(self, weights=None):
        if self.is_multi_device:
            self._weights = RowDistributedTensor.zeros(
                self.device_manager.chunk_sizes(self.size), 
                self.n_outputs, 
                self.dtype, 
                self.device_manager.devices,
                self.device_manager.base_device_idx
            )
        else:
            self._weights = SingleDeviceTensor(torch.zeros(self.size, self.n_outputs, dtype=self.dtype, device=self.device_manager.base_device))

    @property
    def centers(self):
        if self.is_multi_device:
            return slice_distributed_matrix_till_capacity(self._centers, self.used_capacity)
        else:
            return self._centers[:self.used_capacity]
    
    @centers.deleter
    def centers(self):
        del self._centers

    @property
    def weights(self):
        if self.is_multi_device:
            return slice_distributed_matrix_till_capacity(self._weights, self.used_capacity)
        else:
            return self._weights[:self.used_capacity]

    @weights.deleter
    def weights(self):
        del self._weights

    @property
    def used_capacity(self): # of size (num_devices,)
        return self._used_capacity 

    @used_capacity.setter
    def used_capacity(self, new_capacity: torch.Tensor):
        self._used_capacity = new_capacity

    def forward_distributed(self, x, cache_columns_by_idx=None):
        assert_and_raise(x, DistributedTensor)
        assert_and_raise(self.centers, DistributedTensor)
        assert x.num_parts==self.centers.num_parts
        return distributed_kernel_evaluation(self.kernel_fn, x, self.centers) @ self.weights


    def append_centers_and_weights(self, 
            new_centers: Union[torch.Tensor, DistributedTensor], 
            new_weights: Union[torch.Tensor, DistributedTensor]):
        """Adds centers and weights to the kernel machine.
        """

        if self.is_multi_device:
            if (self.used_capacity.sum() + len(new_centers.at_device(self.device_manager.base_device_idx)) > self.size):
                print("error")
                raise ValueError(f"Running out of capacity for new centers: ", self.used_capacity.sum(), new_centers.lengths[0], self.size)
            chunk_sizes = torch.as_tensor(self.device_manager.chunk_sizes(new_centers.lengths[0]))
            chunk_ends = torch.cumsum(chunk_sizes, 0, dtype=torch.int64)
            chunk_starts = chunk_ends - chunk_sizes[0]
            for i in range(len(self.device_manager.devices)):
                self._centers.parts[i][self.used_capacity[i] : self.used_capacity[i] + chunk_sizes[i]] = new_centers.parts[i][chunk_starts[i]:chunk_ends[i]]
                self._weights.parts[i][self.used_capacity[i] : self.used_capacity[i] + chunk_sizes[i]] = new_weights.parts[i][chunk_starts[i]:chunk_ends[i]]
    
            self.used_capacity += chunk_sizes

        else:
            if (self.used_capacity + len(new_centers) > self.size):
                print("error")
                raise ValueError(f"Running out of capacity for new centers: ", self.used_capacity.sum(), len(new_centers), self.size)
            self._centers[self.used_capacity : self.used_capacity + len(new_centers),:] = new_centers
            self._weights[self.used_capacity : self.used_capacity + len(new_weights),:] = new_weights
            self.used_capacity += len(new_centers)

    def reset(self):
        """reset the model to before temporary centers were added.
        """
        self.used_capacity = torch.zeros(
            len(self.device_manager.devices), dtype=torch.int64
        ) if self.is_multi_device else torch.LongTensor([0])
        self.reset_centers()
        self.reset_weights()




class EigenProIterator:
    """EigenPro iterator for general kernel models.
    """

    def __init__(self,
                 model: km.KernelMachine,
                 preconditioner: pcd.Preconditioner = None,
                 state_max_size: int = -1,
                ) -> None:
        """Initialize the EigenPro optimizer."""

        
        self._state_max_size = state_max_size
        self.preconditioner = preconditioner

        self.latent_model = LatentKernelMachine(
            model.kernel_fn, model.n_inputs, model.n_outputs, self.state_max_size,
            dtype=model.dtype, device_manager=model.device_manager, centers=None)

        self.latent_nystrom_model = km.KernelMachine(
            model.kernel_fn, model.n_inputs, model.n_outputs, self.preconditioner.size,
            dtype=model.dtype, device_manager=model.device_manager, 
            # centers=model.device_manager.scatter(self.preconditioner.centers)
            centers=self.preconditioner.centers # they get scattered in km.KernelMachine.__init__()
            )

        if model.is_multi_device:
            self.grad_accumulation = RowDistributedTensor.zeros(
                model.device_manager.chunk_sizes(model.size), 
                model.n_outputs, 
                model.dtype, 
                model.device_manager.devices,
                model.device_manager.base_device_idx
            )
        else:
            self.grad_accumulation = SingleDeviceTensor(torch.zeros(model.size, model.n_outputs, device=model.device_manager.base_device, dtype=model.dtype))
        
        model.eval()
        if model.is_multi_device:
            kmat = distributed_kernel_evaluation(
                        model.kernel_fn,
                        model.device_manager.broadcast(self.preconditioner.centers),
                        model.centers,
                    )
            self.k_centers_nystroms_mult_normalized_eigenvectors = kmat.T @ model.device_manager.broadcast(self.preconditioner.normalized_eigenvectors)
        else:
            self.k_centers_nystroms_mult_normalized_eigenvectors = model.kernel_fn(
                    model.centers, self.preconditioner.centers
                ) @ self.preconditioner.normalized_eigenvectors

        self.preconditioner_normalized_eigenvectors_scattered = model.device_manager.scatter(self.preconditioner.normalized_eigenvectors)
        
        self.projection_dataloader = None

    @property
    def model(self):
        return self._model

    def release_memory_for_projection(self):
        del (
                self.latent_model.centers, 
                self.latent_model.weights, 
                self.latent_nystrom_model.weights
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    @property
    def state_max_size(self):
        return self._state_max_size

    def step(self,
             model: km.KernelMachine,
             batch_x: Union[torch.Tensor, DistributedTensor],
             batch_y: torch.Tensor,
            ) -> None:
        """Performs a single optimization step.

        Args:
            batch_x (torch.Tensor): Batch of input features.
            batch_y (torch.Tensor): Batch of target values.
        """
        assert model._train

        batch_x = model.device_manager.broadcast(batch_x.to(model.dtype))
        batch_y = batch_y.to(model.dtype).to(model.device_manager.base_device)

        batch_p_base = model(batch_x)
        batch_p_temp = self.latent_model(batch_x)
        batch_p_nys = self.latent_nystrom_model(batch_x)

        # gradient in function space K(., batch) (f-y)
        grad = batch_p_base + batch_p_temp + batch_p_nys 
        grad = model.device_manager.reduce_add(grad) - batch_y
        batch_size = grad.shape[0]


        lr = self.preconditioner.scaled_learning_rate(batch_size)

        ### Runs on base device
        kg = model.kernel_fn(self.preconditioner.centers, batch_x.at_device(model.device_manager.base_device_idx)) @ grad
        
        gamma = self.preconditioner.normalized_eigenvectors.T @ kg
        

        ### Broadcast grad and gamma
        gamma = model.device_manager.broadcast(gamma)
        grad = model.device_manager.broadcast(grad)
        
        # Runs on all devices
        self.grad_accumulation -= (model.backward(grad) - self.k_centers_nystroms_mult_normalized_eigenvectors @ gamma)*lr

        self.latent_model.append_centers_and_weights(batch_x, grad*(-lr))

        self.latent_nystrom_model.weights += (self.preconditioner_normalized_eigenvectors_scattered @ gamma)*lr


    def reset(self):
        """Reset the gradient accumulation
        Args:
            None
        return:
            None
        """
        self.latent_model.reset()
        
        self.latent_nystrom_model.init_weights(None)