"""Optimizer class and utility functions for EigenPro iteration."""
import torch
import torch.utils.data as torch_data
import numpy as np

import eigenpro.models.kernel_machine as km
import eigenpro.preconditioners as pcd
import eigenpro.data.array_dataset as array_dataset
import eigenpro.solver.base as base
from eigenpro.utils.types import assert_and_raise
from eigenpro.utils.tensor import DistributedTensor, SingleDeviceTensor, BaseDeviceTensor

def add_inplace_to_part_by_index(
    distributed_tensor: DistributedTensor, 
    device_idx: int, 
    index: SingleDeviceTensor, 
    other_tensor: SingleDeviceTensor, 
    alpha: float):
    assert distributed_tensor.parts[device_idx].device == other_tensor.device
    distributed_tensor.parts[device_idx][index] += alpha*other_tensor

def update_weights_by_index(model, other_tensor, indices, alpha):
    """
    if model.weights are of type DistributedTensor,
    we need `model.weights` indexed by `indices` to be on a single device
    it sends other_tensor to this device and adds it after scaling with alpha
    """
    if not model._train:
        raise ValueError("make sure the input `model` is trainable. Try again after model.train()")
    if model.is_multi_device:
        # assert_and_raise(other_tensor, DistributedTensor)
        # assert_and_raise(indices, SingleDeviceTensor)
        device_idx = np.searchsorted(model.weights._offsets, min(indices))

        add_inplace_to_part_by_index(
            model.weights,
            device_idx,
            indices-model.weights._offsets[device_idx],
            other_tensor,
            alpha
            )

    else:
        # assert_and_raise(other_tensor, SingleDeviceTensor)
        # assert_and_raise(indices, SingleDeviceTensor)
        model.weights[indices] += other_tensor*alpha



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
        self.loader = torch_data.DataLoader(self.dataset, self.preconditioner.critical_batch_size, shuffle=False)


    def step(self,
             batch_gz: torch.Tensor,
             batch_ids: torch.Tensor,
            ):
        lr = self.preconditioner.scaled_learning_rate(len(batch_ids))
        z_batch = self.model.device_manager.broadcast(self.model.centers[batch_ids])
        batch_gz = self.model.device_manager.to_base(batch_gz)
        batch_ids = self.model.device_manager.broadcast(batch_ids)
        gm = self.model.forward(z_batch, cache_columns_by_idx=self.preconditioner.center_ids) 
        
        # do computation while communications occurs
        k = self.model.kernel_fn(self.preconditioner.centers, 
            z_batch.at_device(self.model.device_manager.base_device_idx))
        ftk = self.preconditioner.normalized_eigenvectors.T @ k
        fftk = self.preconditioner.normalized_eigenvectors @ ftk

        gm = self.model.device_manager.reduce_add(gm) - batch_gz

        h = fftk @ gm

        if self.model.is_multi_device:
            batch_device = self.model.weights.get_device_id_by_idx(batch_ids)
            update_weights_by_index(
                self.model, 
                gm, 
                batch_ids.at_device(batch_device), 
                alpha=-lr)
        else:
            update_weights_by_index(self.model, 
                gm, batch_ids, alpha=-lr)
        
        update_weights_by_index(
            self.model, 
            h, self.preconditioner.center_ids, alpha=lr)


