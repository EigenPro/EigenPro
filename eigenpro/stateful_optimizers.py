"""Optimizer class and utility functions for EigenPro iteration."""
import torch

import eigenpro.models.kernel_machine as km
import eigenpro.preconditioners as pcd


class EigenPro:
    """EigenPro optimizer for kernel machines.

    Args:
        model (KernelMachine): A KernelMachine instance.
        threshold_index (int): An index used for thresholding.
        data_preconditioner (Preconditioner): Preconditioner instance that
            contains a top kernel eigensystem for correcting the gradient for
            data.
        model_preconditioner (Preconditioner): Preconditioner instance that
            contains a top kernel eigensystem for correcting the gradient for
            the projection.

    Attributes:
        model (KernelMachine): A KernelMachine instance.
        precon (Preconditioner): A Preconditioner instance.
        _threshold_index (int): An index used for thresholding.
    """

    def __init__(self,
                 model: km.KernelMachine,
                 threshold_index: int,
                 data_preconditioner: pcd.Preconditioner,
                 model_preconditioner: pcd.Preconditioner,
                 kz_xs_evecs:torch.tensor = None,
                 dtype=torch.float32,
                 tmp_centers_coeff : int = 2,
                 accumulated_gradients:bool = True) -> None:
        """Initialize the EigenPro optimizer."""

        self.dtype = dtype
        self._model = model
        self._threshold_index = threshold_index
        self.data_preconditioner  = data_preconditioner
        self.model_preconditioner = model_preconditioner
        
        self.temporary_model_size = int(model.size*tmp_centers_coeff)

        self.temporary_model = km.KernelMachine(
            model.kernel_fn, model.n_outputs, self.temporary_model_size)

        self.nystrom_model = km.KernelMachine(
            model.kernel_fn, model.n_outputs, self.data_preconditioner.size)
        self.nystrom_model.centers = self.data_preconditioner.centers
        
        self.k_centers_nystroms_mult_eigenvecs = self.data_preconditioner.eval_vec(self.model.centers).to(self.dtype)

        self.reset()



    @property
    def model(self) -> km.KernelMachine:
        """Gets the active model (for training).

        Returns:
            KernelMachine: The active model.
        """
        return self._model


    def step(self,
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

        batch_p_base = self.model(batch_x)
        batch_p_temp = self.temporary_model(batch_x)
        batch_p_nys = self.nystrom_model(batch_x)

        # gradient in function space K(bathc,.) (f-y)
        grad = batch_p_base + batch_p_temp + batch_p_nys - batch_y.to(self.dtype).to(batch_p_base.device)
        batch_size = batch_x.shape[0]

        lr = self.data_preconditioner.scaled_learning_rate(batch_size)
        deltap, delta = self.data_preconditioner.delta(
            batch_x.to(grad.device).to(self.dtype), grad.to(self.dtype))

        # k_centers_batch_all = self.model.lru.get('k_centers_batch')
        # self.model.lru.cache.clear()
        # kgrads = []
        # for k in k_centers_batch_all:
        #     kgrads.append(k @ grad.to(k.device).to(k.dtype))
        # k_centers_batch_grad = torch.cat(kgrads)  ##  K(batch, Z) (f-y)
        

        self.grad_accumulation = self.grad_accumulation - lr*\
                                 ( self.model.backward(grad) -
                                   (self.k_centers_nystroms_mult_eigenvecs @
                                    deltap) )

        self.temporary_model.add_centers(batch_x, -lr*grad)
        # print(f'used capacity:{self.model.shard_kms[0].used_capacity}')

        del batch_y


        self.nystrom_model.weights += lr*delta

        del grad, batch_x, batch_p_base, batch_p_temp, batch_p_nys, deltap, delta
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


    def project(self,
             gz: torch.Tensor,
             epochs: int = 1) -> None:
        """Performs a projection.
        """

        batch_p = self.model.forward(batch_x,projection=projection)
        # gradient in function space K(bathc,.) (f-y)
        grad = batch_p - batch_y.to(self.dtype).to(batch_p.device)
        batch_size = batch_x.shape[0]

        lr = self.model_preconditioner.scaled_learning_rate(batch_size)
        deltap, delta = self.model_preconditioner.delta(
            batch_x.to(grad.device).to(self.dtype), grad)

        if self.grad_accumulation is None or projection:
            self.model.update_by_index(batch_ids, -lr*grad,
                                       projection=projection)
            self.model.update_by_index(
                torch.arange(self.model_preconditioner.size),
                lr*delta, nystrom_update=True,
                projection=True
            )

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
        self.grad_accumulation = torch.zeros(self.model.size, self.model.n_outputs, dtype=self.dtype)

        self.temporary_model.centers = torch.zeros(0, self.model.centers.shape[1], dtype=self.dtype)
        self.temporary_model.weights = torch.zeros(0, self.model.n_outputs, dtype=self.dtype)
        
        self.nystrom_model.weights = torch.zeros(self.data_preconditioner.size, self.model.n_outputs, dtype=self.dtype)

