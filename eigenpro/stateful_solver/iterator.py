"""Optimizer class and utility functions for EigenPro iteration."""
import torch

import eigenpro.models.kernel_machine as km
import eigenpro.models.stateful_preallocated_kernel_machine as pkm
import eigenpro.preconditioners as pcd
import eigenpro.stateful_solver.base as base


class EigenProIterator(base.BaseSolver):
    """EigenPro iterator for general kernel models.
    """

    def __init__(self,
                 model: km.KernelMachine,
                 dtype: torch.dtype = torch.float32,
                 preconditioner: pcd.Preconditioner = None,
                 temporary_model_size: int = -1,
                ) -> None:
        """Initialize the EigenPro optimizer."""

        super().__init__(model, dtype)        
        
        self._temporary_model_size = temporary_model_size
        self.preconditioner = preconditioner

        self.temporary_model = pkm.PreallocatedKernelMachine(
            model.kernel_fn, model.n_inputs, model.n_outputs, self.temporary_model_size)

        self.nystrom_model = km.KernelMachine(
            model.kernel_fn, model.n_inputs, model.n_outputs, self.preconditioner.size)

        self.nystrom_model.centers = self.preconditioner.centers
        
        self.k_centers_nystroms_mult_eigenvecs = self.preconditioner.eval_vec(self.model.centers).to(self.dtype)

        self.projection_dataloader = None

    @property
    def temporary_model_size(self):
        return self._temporary_model_size

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
        batch_p_temp = self.temporary_model(batch_x)
        batch_p_nys = self.nystrom_model(batch_x)

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
        self.temporary_model.add_centers(batch_x, -lr*grad)

        self.nystrom_model.weights += lr*delta


    def reset_gradient(self):
        self.grad_accumulation = self.model(self.model.centers) #torch.zeros(self.model.size, self.model.n_outputs, dtype=self.dtype)


    def reset(self):
        """Reset the gradient accumulation
        Args:
            None
        return:
            None
        """
        self.temporary_model.reset()
        
        self.nystrom_model.weights = torch.zeros(self.preconditioner.size, self.model.n_outputs, dtype=self.dtype)