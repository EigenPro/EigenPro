import torch, numpy as np
from .base import KernelMachine
from typing import Callable, List, Optional
from .preallocated_kernel_machine import PreallocatedKernelMachine
from ..utils.cache import LRUCache
from concurrent.futures import ThreadPoolExecutor, as_completed


class ShardedKernelMachine(KernelMachine):
  """Kernel machine that shards its computation across multiple devices."""

  def __init__(self, kms: List[PreallocatedKernelMachine], device: 'Device' ):
    self.device = device
    self.shard_kms = kms
    self.n_devices = len(kms)
    self.n_machines = len(kms)
    self.lru = LRUCache()
    super().__init__(
        kms[0].kernel_fn, 
        kms[0].n_outputs, 
        sum(km.size for km in kms)
    )

  @property
  def centers(self):
    """Return the centers"""
    return [km.centers for km in self.shard_kms]

  @property
  def weights(self):
    """Return the weights"""
    return [km.weights for km in self.shard_kms]



  def forward(self, x: torch.Tensor, projection:bool=False,train=True) -> torch.Tensor:
    """Forward pass for the kernel machine.

    Args:
        x (torch.Tensor): input tensor of shape [n_samples, n_features].
        projection(bool): Projection mode, updating projection weights.
        train(bool): Train mode, storing kernel_mat[:, :self.original_size].T in cache.

    Returns:
        torch.Tensor: tensor of shape [n_samples, n_outputs].
    """

    x_broadcast = self.device(x)
    with ThreadPoolExecutor() as executor:
      predictions = [executor.submit(self.shard_kms[i].forward, x_broadcast[i],
                                     projection=projection,train=train)
                     for i in range(self.n_devices)]
    results = [k.result() for k in predictions]

    p_all = 0
    k_centers_batch_all = []
    for i,r in enumerate(results):
      p_all =  p_all + r.to(self.device.device_base)
      k_centers_batch_all.append(self.shard_kms[i].lru.get('k_centers_batch'))
      self.shard_kms[i].lru.cache.clear()
      torch.cuda.empty_cache()
      del r

    if train:
      self.lru.put('k_centers_batch',k_centers_batch_all )

    del x_broadcast,k_centers_batch_all,results,x
    torch.cuda.empty_cache()


    return p_all

  def add_centers(self, centers: torch.Tensor, weights: Optional[torch.Tensor] = None
                  ,nystrom_centers = False) -> None:
    """Adds new centers and weights.

    Args:
      centers: Kernel centers of shape [n_centers, n_features].
      weights: Weight parameters corresponding to the centers of shape
               [n_centers, n_output].

    Returns:
      center_weights: Weight parameters corresponding to the added centers.
    """
    centers_gpus_list = self.device(centers, strategy="divide_to_gpu")
    center_weights = weights if weights is not None else torch.zeros(
        (centers.shape[0], self.n_outputs))
    weights_gpus_list = self.device(center_weights, strategy="divide_to_gpu")


    with ThreadPoolExecutor() as executor:
      outputs = [executor.submit(self.shard_kms[i].add_centers, centers_gpus_list[i]
                           , weights_gpus_list[i],nystrom_centers=nystrom_centers) for i in range(self.n_devices)]

    for outputs in as_completed(outputs):
      try:
        # Retrieve the result of the future
        _ = outputs.result()
        # Process the result if necessary
      except Exception as exc:
        # Handle exceptions from within the thread
        print(f"A thread caused an error: {exc}")
        raise  # Reraising the exception will stop the program


    return center_weights
  def update_by_index(self, indices: torch.Tensor,
                      delta: torch.Tensor,
                      projection:bool=False,
                      nystrom_update:bool = False) -> None:
    """Update the model weights by index.

    Here we assume that only the first block is trainable.

    Args:
      indices: Tensor of 1-D indices to select rows of weights.
      delta: Tensor of weight update of shape [n_indices, n_outputs].
      projection (bool): update projection weights
      nystrom_update: updating data nystrom samples in the temporory centers phase.

    """
    indices_list = []
    delta_list = []
    threshold_now = 0
    threshold_before = 0

    for i in range(self.n_devices):
      if projection:
        #### only on one gpu, does not work with multi-gpu
        indices_list.append(indices)
        delta_list.append(delta)
      elif nystrom_update:
        number_nystroms_in_gpui = self.shard_kms[i].nystrom_size
        indices_in_gpui = torch.tensor( list(range(self.shard_kms[i].original_size,
                                     self.shard_kms[i].original_size+number_nystroms_in_gpui)
                                             )
                                        )
        indices_list.append(indices_in_gpui)
        delta_list.append(delta[threshold_now:threshold_now+number_nystroms_in_gpui])
        # threshold_now += number_nystroms_in_gpui
      else:
        threshold_now = threshold_now + self.shard_kms[i].original_size #+ self.shard_kms[i].nystrom_size
        gp1_indices = np.where((indices<threshold_now) & (indices>=threshold_before))[0]
        indices_in_gpui = indices[gp1_indices] - threshold_before
        indices_list.append(indices_in_gpui)
        delta_list.append(delta[gp1_indices])
        threshold_before = threshold_now


    with ThreadPoolExecutor() as executor:
      _ = [executor.submit(self.shard_kms[i].update_by_index, indices_list[i],
                           delta_list[i],projection=projection) for i in range(self.n_devices)]


  def reset(self):
    """reset the model to before adding temporary centers were added adn before projection
    Args:
        No arguments.
    Returns:
        None
    """
    with ThreadPoolExecutor() as executor:
      [executor.submit(self.shard_kms[i].reset())
                         for i in range(self.n_devices)]