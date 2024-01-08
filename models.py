"""Classes of Kernel Machines."""

from typing import Callable, List, Optional
import torch
from concurrent.futures import ThreadPoolExecutor,as_completed
from device import Device
from extra import LRUCache

import numpy as np

import ipdb




class KernelMachine:
  """Base class for KernelMachine.
  
  Attributes:
    kernel_fn: A callable function that computes the kernel matrix.
    n_outputs: The number of outputs.
  """
  def __init__(self,
               kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               n_outputs: int):
    self._kernel_fn = kernel_fn
    self._n_outputs = n_outputs
    
  @property
  def kernel_fn(self) -> Callable[[torch.Tensor, torch.Tensor],
                                  torch.Tensor]:
    return self._kernel_fn
  
  @property
  def n_outputs(self) -> int:
    return self._n_outputs

  def __call__(self, *inputs):
    """To add compatibility with other PyTorch models"""
    return self.forward(*inputs)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the kernel machine.
    
    Args:
        x (torch.Tensor): input tensor of shape [n_samples, n_features].
        
    Returns:
        torch.Tensor: tensor of shape [n_samples, n_outputs].
    """
    raise NotImplementedError

  def add_centers(self, centers: torch.Tensor,
                  weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Adds centers and weights to the kernel machine.
    
    Args:
        centers (torch.Tensor): Tensor of shape [n_samples, n_features].
        weights (torch.Tensor, optional): Tensor of shape
          [n_samples, n_outputs].
        
    Returns:
        torch.Tensor: Weight tensor of shape [n_samples, n_outputs].
    """
    raise NotImplementedError
  
  def update_by_index(self, indices: torch.Tensor,
                      delta: torch.Tensor) -> None:
    """Update the model weights by index.
    
    Args:
      indices: Tensor of 1-D indices to select rows of weights.
      delta: Tensor of weight update of shape [n_indices, n_outputs].
    """
    raise NotImplementedError
  
  def shallow_copy(self) -> 'KernelMachine':
    """A dummy shallow copy that returns the current instance."""
    return self



class PreallocatedKernelMachine_optimized(KernelMachine):
  """Kernel machine class for handling kernel methods.

  Attributes:
    n_outputs: The number of outputs.
    n_centers: The number of model centers.
    _centers: A tensor of kernel centers of shape [n_centers, n_features].
    _weights: An optional tensor of weights of shape [n_centers, n_outputs].
  """

  def __init__(
          self,
          kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          n_outputs: int,
          centers: torch.Tensor,
          type= torch.float32,
          tmp_centers_coeff: Optional[int] = 2,
          preallocation_size: Optional[int] = None,
          weights: Optional[torch.Tensor] = None,
          device: Optional[torch.device] = 'cpu') -> None:
    """Initializes the PreallocatedKernelMachine.

    Args:
        kernel_fn (Callable): The kernel function.
        n_outputs (int): The number of outputs.
        centers (torch.Tensor): The tensor of kernel centers.
        type: Data type, e.g. torch.float32 or torch.float16
        tmp_centers_coeff: the ratio between total centers(temporary included) and original centers
        preallocation_size (int, optional): The size for preallocation.
        weights (torch.Tensor, optional): The tensor of weights.
        device (torch.device, optional): The device for tensor operations.
    """
    super().__init__(kernel_fn, n_outputs)

    self.type = type
    self.tmp_centers_coeff = tmp_centers_coeff
    self.device = device
    self.original_size = centers.shape[0]
    self.nystrom_size = 0

    if preallocation_size is None:
      self._centers = torch.zeros(tmp_centers_coeff * centers.shape[0], centers.shape[1],
                                  device=device,dtype=self.type )
      self._weights = torch.zeros(tmp_centers_coeff * centers.shape[0], self._n_outputs,
                                  device=device,dtype=self.type)
    else:
      self._centers = torch.zeros(preallocation_size, centers.shape[1],
                                  device=device,dtype=self.type)
      self._weights = torch.zeros(preallocation_size, self._n_outputs,
                                  device=device,dtype=self.type)

    self.weights_project = torch.zeros(centers.shape[0], self._n_outputs,
                                  device=device,dtype=self.type)


    self.used_capacity = 0
    self.add_centers(centers, weights)

    self.lru = LRUCache()

  @property
  def n_centers(self) -> int:
    """Return the number of centers."""
    return self.used_capacity

  def forward(self, x: torch.Tensor, projection:bool = False, train=True) -> torch.Tensor:
    """Forward pass for the kernel machine.

    Args:
        x (torch.Tensor): input tensor of shape [n_samples, n_features].
        projection(bool): Projection mode, updating projection weights
        train(bool): Train mode, storing kernel_mat[:, :self.original_size].T in cache

    Returns:
        torch.Tensor: tensor of shape [n_samples, n_outputs].
    """

    x = x.to(self.type)
    x = x.to(self.device)

    if projection:
      centers = self._centers[:self.original_size,:]
      weights = self._weights[:self.original_size, :]
    else:
      centers = self._centers[:self.used_capacity, :]
      weights = self._weights[:self.used_capacity, :]

    kernel_mat = self._kernel_fn(x, centers)

    if projection:
      predictions = kernel_mat@self.weights_project
    else:
      poriginal = kernel_mat[:, :self.original_size] @ weights[:self.original_size, :]
      if kernel_mat.shape[1] > self.original_size:
        prest = kernel_mat[:, self.original_size:] @ weights[self.original_size:, :]
      else:
        prest = 0
      predictions = poriginal + prest

      if train:
        self.lru.put('k_centers_batch', kernel_mat[:, :self.original_size].T)

    del x, kernel_mat
    torch.cuda.empty_cache()
    return predictions



  def add_centers(self,
                  centers: torch.Tensor,
                  weights: Optional[torch.Tensor] = None,
                  nystrom_centers: bool = False) -> torch.Tensor:
    """Adds centers and weights to the kernel machine.

    Args:
        centers (torch.Tensor): Tensor of shape [n_samples, n_features].
        weights (torch.Tensor, optional): Tensor of shape
          [n_samples, n_outputs].
        nystrom_centers (bool): True/False.
    Returns:
        None
    """

    if weights is not None:
      weights = weights.to(self.device)
    else:
      weights = torch.zeros((centers.shape[0], self.n_outputs),
                            device=self.device)

    # print(f'capacity used:{self.used_capacity}, newcenters size:{centers.shape[0]}')
    if self.used_capacity + centers.shape[0] > self.tmp_centers_coeff * self.original_size:
      print("error")
      raise ValueError(f"Out of capacity for new centers: "
                       f"{self.used_capacity=} > {self.tmp_centers_coeff} * ({self.original_size=})")

    self._centers[self.used_capacity:self.used_capacity +
                                     centers.shape[0], :] = centers.to(self.type).to(self.device)
    self._weights[self.used_capacity:self.used_capacity +
                                     centers.shape[0], :] = weights
    self.used_capacity += centers.shape[0]

    if nystrom_centers:
      self.nystrom_size = centers.shape[0]

    del centers, weights
    torch.cuda.empty_cache()


  def reset(self):
    """reset the model to before adding temporary centers were added adn before projection

    Args:
        No arguments.
    Returns:
        None
    """
    self.used_capacity = self.original_size + self.nystrom_size
    self._centers[self.original_size + self.nystrom_size:, :] = 0
    self.weights_project = torch.zeros_like(self.weights_project)

    self._weights[self.original_size:, :] = 0


  def update_by_index(self, indices: torch.Tensor,
                      delta: torch.Tensor,projection:bool=False) -> None:
    """Update the model weights by index.

    Args:
      indices: Tensor of 1-D indices to select rows of weights.
      delta: Tensor of weight update of shape [n_indices, n_outputs].
    """
    if projection:
      self.weights_project[indices] +=delta
    else:
      self._weights[indices] += delta




class ShardedKernelMachine(KernelMachine):
  """Kernel machine that shards its computation across multiple devices."""

  def __init__(self, kms: List[PreallocatedKernelMachine_optimized],device: 'Device' ):
    self.device = device
    self.shard_kms = kms
    self.n_devices = len(kms)
    self.n_machines = len(kms)
    self.lru = LRUCache()
    super().__init__(kms[0].kernel_fn, kms[0].n_outputs)



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



class BlockKernelMachine(KernelMachine):
  """Block kernel machine class for handling kernel methods.

  Provides methods for creating and manipulating kernel machines including
  adding centers, computing the forward pass, and copying the current
  instance.

  Attributes:
    n_outputs: The number of outputs.
    kernel_fn: A callable that computes the kernel function.
    _center_blocks: A list of tensors of kernel centers.
    _weight_blocks: A list of trainable tensors of weights.
  """

  def __init__(self,
               kernel_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               n_outputs: int,
               centers: torch.Tensor,
               weights: Optional[torch.Tensor] = None) -> None:
    """Initializes a new KernelMachine instance.

    Args:
      kernel_fn: A callable that computes the kernel function.
      n_outputs: The number of outputs.
      centers: A tensor of kernel centers of shape [n_centers, n_features].
      weights: An optional tensor of weights of shape [n_centers, n_outputs].
    """
    super().__init__(kernel_fn, n_outputs)
    self.original_size = centers.shape[0]
    self._kernel_fn = kernel_fn
    self._n_outputs = n_outputs
    self._center_blocks = []
    self._weight_blocks = []
    self.lru = LRUCache()
    _ = self.add_centers(centers, weights)

  @property
  def n_outputs(self) -> int:
    """Returns the number of outputs."""
    return self._n_outputs

  @property
  def n_centers(self) -> int:
    """Returns the number of centers."""
    count = 0
    for center_block in self._center_blocks:
      count += center_block.size(dim=0)
    return count

  @property
  def weights(self) -> torch.Tensor:
    """Returns all weights as a single Tensor."""
    return torch.cat(self._weight_blocks, dim=0)

  def update_by_index(self, indices: torch.Tensor,
                      delta: torch.Tensor) -> None:
    """Update the model weights by index.

    Here we assume that only the first block is trainable.

    Args:
      indices: Tensor of 1-D indices to select rows of weights.
      delta: Tensor of weight update of shape [n_indices, n_outputs].
    """
    self._weight_blocks[0][indices] += delta

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass of the kernel model.

    Args:
      x: Input mini-batch tensor of shape [batch_size, n_features].

    Returns:
      Output tensor of shape [batch_size, n_outputs].
    """
    # Center matrix of shape [n_centers, n_features].
    centers = torch.cat(self._center_blocks, dim=0)
    # Kernel matrix of shape [batch_size, n_centers].
    kernel_mat = self._kernel_fn(x, centers)
    # cache teh matrix
    self.lru.put('k_centers_batch_grad', kernel_mat[:, :self.original_size])
    # Weight matrix of shape [n_centers, n_outputs].
    weights = torch.cat(self._weight_blocks, dim=0)
    # Output of shape [batch_size, n_outputs]
    output = kernel_mat.matmul(weights)
    return output

  def add_centers(self,
                  centers: torch.Tensor,
                  weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Adds new centers and weights.

    Args:
      centers: Kernel centers of shape [n_centers, n_features].
      weights: Weight parameters corresponding to the centers of shape
               [n_centers, n_output].

    Returns:
      center_weights: Weight parameters corresponding to the added centers.
    """
    self._center_blocks.append(centers)
    # TODO: Check if the number of rows of centers equals that of weights.
    center_weights = weights if weights is not None else torch.zeros(
      (centers.shape[0], self.n_outputs))
    self._weight_blocks.append(center_weights)
    return center_weights

  def shallow_copy(self) -> 'KernelMachine':
    """Creates a shallow copy of the current kernel machine.

    Returns:
      A new KernelMachine instance that shares the same underlying centers and
      weights.
    """
    copy = BlockKernelMachine(self._kernel_fn, self.n_outputs,
                              self._center_blocks[0], self._weight_blocks[0])
    for center_block, weight_block in zip(self._center_blocks[1:],
                                          self._weight_blocks[1:]):
      copy.add_centers(center_block, weight_block)
    return copy


