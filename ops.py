from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Any
# Assuming PyTorch's tensors are used, you should import torch as well
# import torch

class ParallelComputation:
    """
    This class contains methods for performing parallel computations,
    specifically for matrix operations and kernel evaluations.
    """

    @staticmethod
    def compute_kernel_matrix(kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                              batch_x: List['torch.Tensor'],chunks_z: List['torch.Tensor']) -> List['torch.Tensor']:
        """
        This method evaluates the given kernel function in parallel for each pair of batch_x and chunks_z.
        Each item in batch_x and each chunk in chunks_z could potentially reside on different devices.
        """
        with ThreadPoolExecutor() as executor:
            kernel_matrices = [executor.submit(kernel, inputs[0], inputs[1]) for inputs in zip(batch_x, chunks_z)]

        computed_kernel_matrices = [job.result() for job in kernel_matrices]

        return computed_kernel_matrices

    @staticmethod
    def compute_matrix_vector_product(matrix: 'torch.Tensor', vector: 'torch.Tensor', device: str) -> 'torch.Tensor':
        """
        This method multiplies the given matrix with the vector.
        """
        return (matrix @ vector).to(device)

    @staticmethod
    def compute_parallel_matrix_vector_product(matrix_list: List['torch.Tensor'],
                                               vector_list: List['torch.Tensor'], device: str) -> 'torch.Tensor':
        """
        This method performs matrix-vector multiplication operation in parallel.
        """
        with ThreadPoolExecutor() as executor:
            products = [executor.submit(ParallelComputation.compute_matrix_vector_product, inputs[0], inputs[1], device)
                        for inputs in zip(matrix_list, vector_list)]

        sum_of_products = sum(job.result() for job in products)

        return sum_of_products
