"""Optimizer class and utility functions for EigenPro iteration."""
import torch

import eigenpro.models.kernel_machine as km
import eigenpro.preconditioners as pcd
import eigenpro.stateful_solver.base as base


from typing import Callable, List, Optional
import torch

import eigenpro.models.kernel_machine as km
import eigenpro.utils.cache as cache
import eigenpro.utils.fmm as fmm


class LatentKernelMachine(km.KernelMachine):
    """Kernel machine class for handling kernel methods.
    """

    def __init__(self, *args, **kwargs):
        """Initializes a KernelMachine with fixed size.
        """
        super().__init__(*args, **kwargs)

        self.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the kernel machine.
        Args:
                x (torch.Tensor): input tensor of shape [n_samples, n_inputs].
        Returns:
                torch.Tensor: tensor of shape [n_samples, n_outputs].
        """
        return fmm.KmV(self.kernel_fn, x, 
            self.centers[:self.used_capacity],
            self.weights[:self.used_capacity], col_chunk_size=2**16)


    def append_centers_and_weights(self, new_centers: torch.Tensor, new_weights: torch.Tensor):
        """Adds centers and weights to the kernel machine.
        """

        if (self.used_capacity + len(new_centers) > self.size):
            print("error")
            raise ValueError(f"Running out of capacity for new centers: ")

        self._centers[self.used_capacity: self.used_capacity + len(new_centers),:] = new_centers
        self._weights[self.used_capacity: self.used_capacity + len(new_weights),:] = new_weights
        self.used_capacity += len(new_centers)


    def reset(self):
        """reset the model to before temporary centers were added.
        """
        self.used_capacity = 0
        self.centers = torch.zeros(self.size, self.n_inputs, device=self.device, dtype=self.dtype)
        self.weights = torch.zeros(self.size, self.n_outputs, device=self.device, dtype=self.dtype)




class EigenProIterator(base.BaseSolver):
    """EigenPro iterator for general kernel models.
    """

    def __init__(self,
                 model: km.KernelMachine,
                 dtype: torch.dtype = torch.float32,
                 preconditioner: pcd.Preconditioner = None,
                 state_max_size: int = -1,
                ) -> None:
        """Initialize the EigenPro optimizer."""

        super().__init__(model, dtype)        
        
        self._state_max_size = state_max_size
        self.preconditioner = preconditioner

        self.latent_model = LatentKernelMachine(
            model.kernel_fn, model.n_inputs, model.n_outputs, self.state_max_size,
            dtype=model.dtype, device=model.device)

        self.latent_nystrom_model = km.KernelMachine(
            model.kernel_fn, model.n_inputs, model.n_outputs, self.preconditioner.size,
            dtype=model.dtype, device=model.device)

        self.latent_nystrom_model.centers = self.preconditioner.centers

        self.grad_accumulation = torch.zeros(model.size, model.n_outputs)
        
        self.k_centers_nystroms_mult_eigenvecs = self.preconditioner.eval_vec(self.model.centers).to(self.dtype)

        self.projection_dataloader = None

    @property
    def state_max_size(self):
        return self._state_max_size

    def step(self,
             batch_x: torch.Tensor,
             batch_y: torch.Tensor,
            ) -> None:
        """Performs a single optimization step.

        Args:
            batch_x (torch.Tensor): Batch of input features.
            batch_y (torch.Tensor): Batch of target values.
        """

        batch_p_base = self.model(batch_x)
        batch_p_temp = self.latent_model(batch_x)
        batch_p_nys = self.latent_nystrom_model(batch_x)

        # gradient in function space K(., batch) (f-y)
        grad = batch_p_base + batch_p_temp + batch_p_nys - batch_y.to(self.dtype).to(batch_p_base.device)
        batch_size = batch_x.shape[0]

        lr = self.preconditioner.scaled_learning_rate(batch_size)
        deltap, delta = self.preconditioner.delta(
            batch_x.to(grad.device).to(self.dtype), grad.to(self.dtype))

        self.grad_accumulation -= lr*\
                                 ( self.model.backward(grad) -
                                   (self.k_centers_nystroms_mult_eigenvecs @
                                    deltap) )
        self.latent_model.append_centers_and_weights(batch_x, -lr*grad)

        self.latent_nystrom_model.weights += lr*delta


    def reset_gradient(self):
        self.grad_accumulation = self.model(self.model.centers) #torch.zeros(self.model.size, self.model.n_outputs, dtype=self.dtype)


    def reset(self):
        """Reset the gradient accumulation
        Args:
            None
        return:
            None
        """
        self.latent_model.reset()
        
        self.latent_nystrom_model.weights = torch.zeros(self.preconditioner.size, self.model.n_outputs, dtype=self.dtype)