"""Optimizer class and utility functions for EigenPro iteration."""
import torch
import math
import torch.utils.data as torch_data
from itertools import cycle
import numpy as np
from typing import Union

import eigenpro.kernel_machine as km
import eigenpro.preconditioner as pcd
import eigenpro.data.array_dataset as array_dataset
from eigenpro.utils.types import assert_and_raise
from eigenpro.utils.tensor import DistributedTensor, SingleDeviceTensor, BaseDeviceTensor


def update_weights_by_index(model, other_tensor, indices, alpha):
    """
    if model.weights are of type DistributedTensor,
    we need `model.weights` indexed by `indices` to be on a single device
    it sends other_tensor to this device and adds it after scaling with alpha
    """
    if not model._train:
        raise ValueError("make sure the input `model` is trainable. Try again after model.train()")
    if model.is_multi_device:
        device_idx = get_device_id_by_idx(model.weights, indices)
        indices_for_device = indices - model.weights.offsets[device_idx]
        model_weights_part = model.weights.parts[device_idx]
        model_weights_part[indices_for_device] += alpha*other_tensor.to(model_weights_part.device)
    else:
        assert_and_raise(other_tensor, torch.Tensor)
        assert_and_raise(indices, torch.Tensor)
        model.weights[indices] += other_tensor*alpha

def get_device_id_by_idx(distributed_tensor: DistributedTensor, index: Union[int, SingleDeviceTensor]):
    device_idx = torch.searchsorted(distributed_tensor.offsets, index, right=True) - 1
    unique_device = torch.unique(device_idx)
    assert len(unique_device)==1
    return unique_device


class EigenProProjector:
    """EigenPro optimizer for classical kernel models.
    """
    def __init__(self, 
            model: km.KernelMachine,
            preconditioner: pcd.Preconditioner,
        ):
        self.preconditioner = preconditioner
        self.dataset = array_dataset.ArrayDataset(model.centers, 
            torch.zeros(model.size, model.n_inputs, dtype=model.dtype, device=model.device_manager.base_device))
        pd = model.size//len(model.device_manager.devices)
        batch_size = pd//math.ceil(pd/self.preconditioner.critical_batch_size)
        # make sure batch size is divisible by model_size//num_devices
        self.loader = torch_data.DataLoader(self.dataset, batch_size, shuffle=False)

        # creating a dataloader for projection
        # _batch_device_ids = repeat(range(len(model.device_manager.devices)), )
        # _offsets = 0 if not isinstance(model.weights, DistributedTensor) else model.weights.offsets
        # _chunk_sizes = model.size if not isinstance(model.weights, DistributedTensor) else model.device_manager.chunk_sizes(model.size)
        # idx_per_device = torch.arange(model.size).split(_chunk_sizes.tolist())
        # batch_ids = [idx[torch.randperm(c)].split(self.preconditioner.critical_batch_size) 
        #     for c in _chunk_sizes
        #     for idx in idx_per_device]
        # batch_sampler = (torch.randint(
        #         low=_offsets[b],
        #         high=_offsets[b]+_chunk_sizes[b],
        #         size=(self.preconditioner.critical_batch_size,)
        #     ) for b in _batch_device_ids)
        # self.loader = torch_data.DataLoader(self.dataset, batch_sampler=batch_ids)


    def step(self,
             model,
             batch_gz: torch.Tensor,
             batch_ids: torch.Tensor,
            ):
        assert model._train
        lr = self.preconditioner.scaled_learning_rate(len(batch_ids))
        z_batch = model.device_manager.broadcast(model.centers[batch_ids])
        # batch_gz = model.device_manager.to_base(batch_gz)
        gm = model(z_batch, cache_columns_by_idx=self.preconditioner.center_ids)

        
        # do computation while communications occurs
        k = model.kernel_fn(self.preconditioner.centers, 
            z_batch.at_device(model.device_manager.base_device_idx))
        ftk = self.preconditioner.normalized_eigenvectors.T @ k
        fftk = self.preconditioner.normalized_eigenvectors @ ftk

        
        gm = model.device_manager.reduce_add(gm)
        
        gm = gm - batch_gz

        h = fftk @ gm

        if model.is_multi_device:
            # print(batch_ids, batch_ids.at_device(batch_device))
            # update_weights_by_index(model, gm, batch_ids.at_device(batch_device), alpha=-lr)
            update_weights_by_index(model, gm, batch_ids, alpha=-lr)
        else:
            update_weights_by_index(model, gm, batch_ids, alpha=-lr)
        
        update_weights_by_index(model, h, self.preconditioner.center_ids, alpha=lr)


