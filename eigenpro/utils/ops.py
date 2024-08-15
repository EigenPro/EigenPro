from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor
import torch


class ParallelMatrixOperator:
    """Parallel computations for matrix and kernel operations."""

    @staticmethod
    def compute_kernel_matrix(
        kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_x: List[torch.Tensor],
        chunks_z: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Evaluates a kernel function in parallel.

        Args:
            kernel: The kernel function to evaluate.
            batch_x: List of input tensors.
            chunks_z: List of chunk tensors.

        Returns:
            List of computed kernel matrices.
        """
        with ThreadPoolExecutor() as executor:
            kernel_matrices = [
                executor.submit(kernel, inputs[0], inputs[1])
                for inputs in zip(batch_x, chunks_z)
            ]

        return [job.result() for job in kernel_matrices]

    @staticmethod
    def mat_vec_mul(
        matrix: torch.Tensor, vector: torch.Tensor, device: str
    ) -> torch.Tensor:
        """Multiplies a matrix with a vector.

        Args:
            matrix: Input matrix.
            vector: Input vector.
            device: Device for the output tensor.

        Returns:
            Result of the multiplication.
        """
        return (matrix @ vector).to(device)

    @staticmethod
    def parallel_mat_vec_mul(
        matrix_list: List[torch.Tensor], vector_list: List[torch.Tensor], device: str
    ) -> torch.Tensor:
        """Performs matrix-vector multiplication in parallel.

        Args:
            matrix_list: List of input matrices.
            vector_list: List of input vectors.
            device: Device for the output tensor.

        Returns:
            Sum of the multiplication results.
        """
        with ThreadPoolExecutor() as executor:
            products = [
                executor.submit(
                    ParallelMatrixOperator.mat_vec_mul, inputs[0], inputs[1], device
                )
                for inputs in zip(matrix_list, vector_list)
            ]

        return sum(job.result() for job in products)
