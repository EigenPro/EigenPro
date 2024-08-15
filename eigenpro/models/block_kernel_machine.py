from typing import Callable, Optional
import torch

import eigenpro.models.kernel_machine as km


class BlockKernelMachine(km.KernelMachine):
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
    super().__init__(kernel_fn, n_outputs, centers.shape[0])
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