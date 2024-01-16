"""Optimizer class and utility functions for EigenPro iteration."""
import torch
from .models import KernelMachine
from .preconditioner import Preconditioner

import ipdb

class EigenPro:
    """EigenPro optimizer for kernel machines.

    Args:
        model (KernelMachine): A KernelMachine instance.
        threshold_index (int): An index used for thresholding.
        precon_data (Preconditioner): Preconditioner instance that contains a
            top kernel eigensystem for correcting the gradient for data.
        precon_model (Preconditioner): Preconditioner instance that contains a
            top kernel eigensystem for correcting the gradient for the projection

    Attributes:
        model (KernelMachine): A KernelMachine instance.
        precon (Preconditioner): A Preconditioner instance.
        _threshold_index (int): An index used for thresholding.
    """

    def __init__(self,
                 model: KernelMachine,
                 threshold_index: int,
                 precon_data: Preconditioner,
                 precon_model: Preconditioner,
                 kz_xs_evecs:torch.tensor = None,
                 type=torch.float32,
                 accumulated_gradients:bool = False,) -> None:
        """Initialize the EigenPro optimizer."""

        self.type = type
        self._model = model
        self._threshold_index = threshold_index
        self.precon_data  = precon_data
        self.precon_model = precon_model

        if accumulated_gradients:
            self.grad_accumulation = 0
            if kz_xs_evecs ==None:
                raise NotImplementedError
            else:
                self.k_centers_nystroms_mult_eigenvecs = kz_xs_evecs
        else:
            self.grad_accumulation = None

        #### adding nystrom samples to the model
        self._model.add_centers(precon_data.centers.to(type), None,nystrom_centers = True)



    @property
    def model(self) -> KernelMachine:
        """Gets the active model (for training).

        Returns:
            KernelMachine: The active model.
        """
        return self._model


    def step(self,
             batch_x: torch.Tensor,
             batch_y: torch.Tensor,
             batch_ids: torch.Tensor,
             projection:bool=False) -> None:
        """Performs a single optimization step.

        Args:
            batch_x (torch.Tensor): Batch of input features.
            batch_y (torch.Tensor): Batch of target values.
            batch_ids (torch.Tensor): Batch of sample indices.
            projection (bool): projection mode
        """

        batch_p = self.model.forward(batch_x,projection=projection)
        grad = batch_p - batch_y.to(self.type).to(batch_p.device) ## gradient in function space K(bathc,.) (f-y)
        batch_size = batch_x.shape[0]


        if projection:
            lr = self.precon_model.scaled_learning_rate(batch_size)
            deltap, delta = self.precon_model.delta(batch_x.to(grad.device).to(self.type), grad)
        else:
            lr = self.precon_data.scaled_learning_rate(batch_size)
            deltap, delta = self.precon_data.delta(batch_x.to(grad.device).to(self.type), grad)

        if self.grad_accumulation is None or projection:
            self.model.update_by_index(batch_ids, -lr *grad,projection=projection )
        else:
            k_centers_batch_all = self.model.lru.get('k_centers_batch')
            self.model.lru.cache.clear()
            kgrads = []
            for k in k_centers_batch_all:
                kgrads.append((k @ grad.to(k.device)))
            k_centers_batch_grad = torch.cat(kgrads)  ##  K(bathc,Z) (f-y)

            self.grad_accumulation = self.grad_accumulation - lr*\
                                     ( k_centers_batch_grad -
                                       (self.k_centers_nystroms_mult_eigenvecs @
                                        deltap) )

            self.model.add_centers(batch_x, -lr*grad)
            # print(f'used capacity:{self.model.shard_kms[0].used_capacity}')

            del k_centers_batch_grad, kgrads, k_centers_batch_all,batch_y




        self.model.update_by_index(torch.tensor(list(range(self.precon_model._centers.shape[0])))
                                   ,lr*delta, nystrom_update=True,projection=projection)

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
