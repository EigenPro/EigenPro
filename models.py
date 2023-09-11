"""Classes of Kernel Machines."""

from typing import Callable, List, Optional
import torch
from concurrent.futures import ThreadPoolExecutor
from device import Device


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


class PreallocatedKernelMachine(KernelMachine):
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
    preallocation_size: Optional[int] = None,
    weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = 'cpu') -> None:
    """Initializes the PreallocatedKernelMachine.
    
    Args:
        kernel_fn (Callable): The kernel function.
        n_outputs (int): The number of outputs.
        centers (torch.Tensor): The tensor of kernel centers.
        preallocation_size (int, optional): The size for preallocation.
        weights (torch.Tensor, optional): The tensor of weights.
        device (torch.device, optional): The device for tensor operations.
    """
    super().__init__(kernel_fn, n_outputs)
    self.device = device
    self.original_size = centers.shape[0]
    if preallocation_size is None:
      self._centers = torch.zeros(2 * centers.shape[0], centers.shape[1],
                                  device=device)
      self._weights = torch.zeros(2 * centers.shape[0], self._n_outputs,
                                  device=device)
    else:
      self._centers = torch.zeros(preallocation_size, centers.shape[1],
                                  device=device)
      self._weights = torch.zeros(preallocation_size, self._n_outputs,
                                  device=device)
    self.used_capacity = 0
    self.add_centers(centers, weights)

  @property
  def n_centers(self) -> int:
    """Return the number of centers."""
    return self.used_capacity

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the kernel machine.
    
    Args:
        x (torch.Tensor): input tensor of shape [n_samples, n_features].
        
    Returns:
        torch.Tensor: tensor of shape [n_samples, n_outputs].
    """
    x = x.to(self.device)
    kernel_mat = self._kernel_fn(x, self._centers[:self.used_capacity, :])
    weights = self._weights[:self.used_capacity, :]
    return kernel_mat @ weights
    
  def add_centers(self,
                  centers: torch.Tensor,
                  weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Adds centers and weights to the kernel machine.
    
    Args:
        centers (torch.Tensor): Tensor of shape [n_samples, n_features].
        weights (torch.Tensor, optional): Tensor of shape
          [n_samples, n_outputs].
        
    Returns:
        torch.Tensor: Weight tensor of shape [n_samples, n_outputs].
    """
    centers = centers.to(self.device)
    if weights is not None:
        weights = weights.to(self.device)
    else:
        weights = torch.zeros((centers.shape[0], self.n_outputs),
                              device=self.device)

    self._centers[self.used_capacity:self.used_capacity +
                  centers.shape[0], :] = centers
    self._weights[self.used_capacity:self.used_capacity +
                  centers.shape[0], :] = weights
    self.used_capacity += centers.shape[0]

    if self.used_capacity > 2 * self.original_size:
      raise ValueError(f"Out of capacity for new centers: "
                       f"{self.used_capacity=} > 2 * ({self.original_size=})")
    return weights
  
  def update_by_index(self, indices: torch.Tensor,
                      delta: torch.Tensor) -> None:
    """Update the model weights by index.
    
    Args:
      indices: Tensor of 1-D indices to select rows of weights.
      delta: Tensor of weight update of shape [n_indices, n_outputs].
    """
    self._weights[indices] += delta


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
    self._kernel_fn = kernel_fn
    self._n_outputs = n_outputs
    self._center_blocks = []
    self._weight_blocks = []
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



class ShardedKernelMachine(KernelMachine):
  """Kernel machine that shards its computation across multiple devices."""

  def __init__(self, kms: List[PreallocatedKernelMachine]):
    self.shard_kms = kms
    self.n_machines = len(kms)
    super().__init__(kms[0].kernel_fn, kms[0].n_outputs)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for the kernel machine.
    
    Args:
        x (torch.Tensor): input tensor of shape [n_samples, n_features].
        
    Returns:
        torch.Tensor: tensor of shape [n_samples, n_outputs].
    """
    with ThreadPoolExecutor() as executor:
        kernel_matrices = [executor.submit(self.shard_kms[i], x)
                            for i in range(self.n_machines)]
    return [k.result() for k in kernel_matrices]

  def add_centers(self, centers_list: torch.Tensor) -> None:
    with ThreadPoolExecutor() as executor:
        _ = [executor.submit(self.shard_kms[i].add_centers, centers_list[i]) for
              i in range(self.n_machines)]
