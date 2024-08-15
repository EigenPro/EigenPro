"""Optimizer class and utility functions for EigenPro iteration."""

import torch

import eigenpro.models.kernel_machine as km
import eigenpro.preconditioner as pcd


class EigenPro:
    """EigenPro optimizer for kernel machines.

    Args:
        model (KernelMachine): A KernelMachine instance.
        data_preconditioner (Preconditioner): Preconditioner instance that
            contains a top kernel eigensystem for correcting the gradient for
            data.


    Attributes:
        model (KernelMachine): A KernelMachine instance.
        precon (Preconditioner): A Preconditioner instance.
    """

    def __init__(
        self,
        model: km.KernelMachine,
        data_preconditioner: pcd.Preconditioner,
        kz_xs_evecs: torch.tensor = None,
        dtype=torch.float32,
        accumulated_gradients: bool = False,
    ) -> None:
        """Initialize the EigenPro optimizer."""

        self.dtype = dtype
        self._model = model
        self.data_preconditioner = data_preconditioner

        if accumulated_gradients:
            self.grad_accumulation = 0
            if kz_xs_evecs == None:
                raise NotImplementedError
            else:
                self.k_centers_nystroms_mult_eigenvecs = kz_xs_evecs
        else:
            self.grad_accumulation = None

        # Initilizing nystrom samples to the model
        self._model.init_nystorm(data_preconditioner.centers.to(dtype))

    @property
    def model(self) -> km.KernelMachine:
        """Gets the active model (for training).

        Returns:
            KernelMachine: The active model.
        """
        return self._model

    def step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_ids: torch.Tensor,
    ) -> None:
        """Performs a single optimization step.

        Args:
            batch_x (torch.Tensor): Batch of input features.
            batch_y (torch.Tensor): Batch of target values.
            batch_ids (torch.Tensor): Batch of sample indices.
        """

        batch_p = self.model.forward(batch_x)
        # gradient in function space K(bathc,.) (f-y)
        grad = batch_p - batch_y.to(self.dtype).to(batch_p.device)
        batch_size = batch_x.shape[0]

        lr = self.data_preconditioner.scaled_learning_rate(batch_size)
        deltap, delta = self.data_preconditioner.delta(
            batch_x.to(grad.device).to(self.dtype), grad.to(self.dtype)
        )

        if self.grad_accumulation is None:
            self.model.update_by_index(batch_ids, -lr * grad)
        else:
            k_centers_batch_all = self.model.lru.get("k_centers_batch")
            self.model.lru.cache.clear()
            kgrads = []
            for k in k_centers_batch_all:
                kgrads.append(k @ grad.to(k.device).to(k.dtype))
            k_centers_batch_grad = torch.cat(kgrads)  ##  K(bathc,Z) (f-y)

            self.grad_accumulation = self.grad_accumulation - lr * (
                k_centers_batch_grad - (self.k_centers_nystroms_mult_eigenvecs @ deltap)
            )

            self.model.add_centers(batch_x, -lr * grad)
            # print(f'used capacity:{self.model.shard_kms[0].used_capacity}')

            del k_centers_batch_grad, kgrads, k_centers_batch_all, batch_y

        self.model.shard_kms[0].update_nystroms(lr * delta)

        del grad, batch_x, batch_p, deltap, delta
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def reset(self):
        """Reset the gradient accumulation
        Args:
            None
        return:
            None
        """
        self.grad_accumulation = 0
