"""Optimizer class and utility functions for EigenPro iteration."""
import torch
import torch.utils.data as torch_data

import eigenpro.models.kernel_machine as km
import eigenpro.preconditioners as pcd
import eigenpro.data.array_dataset as array_dataset
import eigenpro.stateful_solver.base as base


class EigenProProjector(base.BaseSolver):
    """EigenPro optimizer for classical kernel models.
    """
    def __init__(self, 
            model: km.KernelMachine,
            dtype: torch.dtype,
            preconditioner: pcd.Preconditioner,
        ):
        
        super().__init__(model, dtype)
        self.preconditioner = preconditioner
        self.dataset = array_dataset.ArrayDataset(self.model.centers, torch.zeros(self.model.size, self.model.n_inputs))
        self.loader = torch_data.DataLoader(self.dataset, self.preconditioner.critical_batch_size, shuffle=True)


    def step(self,
             batch_gz: torch.Tensor,
             batch_ids: torch.Tensor,
            ):
        lr = self.preconditioner.scaled_learning_rate(len(batch_ids))
        z_batch = self.model.device_manager.broadcast(self.model.centers[batch_ids])
        batch_gz = self.model.device_manager.broadcast(batch_gz)
        batch_ids = self.model.device_manager.broadcast(batch_ids)
        gm = self.model.forward(z_batch, cache_columns_by_idx=self.preconditioner.center_ids) - batch_gz
        gm = self.model.device_manager.reduce_add(gm)

        self.model.update_weights_by_index(-lr*gm, batch_ids)

        h = self.model.backward(gm)
        fth = self.preconditioner.normalized_eigenvectors.T @ h
        ffth = self.preconditioner.normalized_eigenvectors @ fth
        self.model.update_weights_by_index(lr*ffth, self.preconditioner.center_ids)


class DistributedEigenProProjector(base.BaseSolver):
    """EigenPro optimizer for classical kernel models.
    """
    def __init__(self, 
            model: km.KernelMachine,
            dtype: torch.dtype,
            preconditioner: pcd.Preconditioner,
        ):
        
        super().__init__(model, dtype)
        self.preconditioner = preconditioner
        self.dataset = array_dataset.ArrayDataset(self.model.centers, torch.zeros(self.model.size, self.model.n_inputs))
        self.loader = torch_data.DataLoader(self.dataset, self.preconditioner.critical_batch_size, shuffle=True)


    def step(self,
             batch_gz: torch.Tensor,
             batch_ids: torch.Tensor,
            ):
        lr = self.preconditioner.scaled_learning_rate(len(batch_ids))

        gm = self.model.forward(self.model.centers[batch_ids], cache_columns_by_idx=self.preconditioner.center_ids) - batch_gz
        self.model.update_weights_by_index(batch_ids, -lr*gm)

        h = self.model.backward(gm)
        fth = self.preconditioner.normalized_eigenvectors.T @ h
        ffth = self.preconditioner.normalized_eigenvectors @ fth
        self.model.update_weights_by_index(self.preconditioner.center_ids, lr*ffth)