"""Optimizer class and utility functions for EigenPro iteration."""
import torch
import torch.utils.data as torch_data

import eigenpro.models.kernel_machine as km
import eigenpro.models.stateful_preallocated_kernel_machine as pkm
import eigenpro.preconditioners as pcd
import eigenpro.data.array_dataset as array_dataset



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

        self.temporary_model = pkm.PreallocatedKernelMachine(
            model.kernel_fn, model.n_inputs, model.n_outputs, self.temporary_model_size)

        self.nystrom_model = km.KernelMachine(
            model.kernel_fn, model.n_inputs, model.n_outputs, self.data_preconditioner.size)

        self.nystrom_model.centers = self.data_preconditioner.centers
        
        self.k_centers_nystroms_mult_eigenvecs = self.data_preconditioner.eval_vec(self.model.centers).to(self.dtype)

        self.reset()

        self.projection_dataloader = None


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

        del batch_y


        self.nystrom_model.weights += lr*delta

        del grad, batch_x, batch_p_base, batch_p_temp, batch_p_nys, deltap, delta
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


    def create_projection_dataloader(gz: torch.Tensor, batch_size: int = None):
        if self.projection_dataloader is None:
            projection_dataset = array_dataset.ArrayDataset(self.model.centers, self.grad_accumulation)
            self.projection_dataloader = torch_data.DataLoader(
                projection_dataset, batch_size=model_batch_size, shuffle=True)
        else:
            self.projection_dataloader


    def project(self, epochs: int = 1, batch_size) -> None:
        """Performs a projection using EigenPro2.
        """
        self.model.train()
        weights_before_projection = self.model.weights
        model_nystrom_ids = torch.arange(self.model_preconditioner.size)
        for _ in range(epochs):
            for z_batch, grad_batch, batch_ids in tqdm(
                    projection_loader, 
                    total=len(projection_loader), 
                    leave=False,
                    desc="Projection"
                ):
                lr = self.model_preconditioner.scaled_learning_rate(len(batch_ids))

                gm = self.model(z_batch) - grad_batch
                self.model.update_weights_by_index(batch_ids, -lr*gm)

                h = self._kmat_batch_centers_cached[:, model_nystrom_ids].T @ gm
                fth = self.model_preconditioner.normalized_eigenvectors.T @ h
                ffth = self.model_preconditioner.normalized_eigenvectors @ fth
                self.model.update_weights_by_index(model_nystrom_ids, lr*ftfh)

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

        self.temporary_model.centers = torch.zeros(
            self.temporary_model_size, self.model.n_inputs, dtype=self.dtype, device=self.temporary_model.device)
        self.temporary_model.weights = torch.zeros(
            self.temporary_model_size, self.model.n_outputs, dtype=self.dtype, device=self.temporary_model.device)
        
        self.nystrom_model.weights = torch.zeros(self.data_preconditioner.size, self.model.n_outputs, dtype=self.dtype)

