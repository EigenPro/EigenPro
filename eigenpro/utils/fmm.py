import torch
import math


def KmV(kernel_f, X, Z, v, out=None, row_chunk_size=None, col_chunk_size=None):
    """Computes the kernel matrix-vector product K(X, Z) @ v.

    If the `out` argument is provided, the result is added to `out`. The
    computation does not store the kernel matrix

    Args:
        kernel_f: A function representing the kernel.
        X: A tensor representing the left input to the kernel.
        Z: A tensor representing the right input to the kernel.
        v: A vector to be multiplied by the kernel matrix.
        out: An optional tensor to store the result. If not provided, a new
            tensor is created.
        row_chunk_size: An optional integer specifying the chunk size for rows.
            If not provided, no chunking is done.
        col_chunk_size: An optional integer specifying the chunk size for
            columns. If not provided, no chunking is done.

    Returns:
        A tensor containing the result of the kernel matrix-vector product.
    """
    n_r, n_c = len(X), len(Z)
    b_r = n_r if row_chunk_size is None else row_chunk_size
    b_c = n_c if col_chunk_size is None else col_chunk_size
    if out is None:
        out = torch.zeros(n_r, *v.shape[1:], device=v.device)

    for i in range(math.ceil(n_r / b_r)):
        for j in range(math.ceil(n_c / b_c)):
            out[i * b_r : (i + 1) * b_r] += (
                kernel_f(X[i * b_r : (i + 1) * b_r], Z[j * b_c : (j + 1) * b_c])
                @ v[j * b_c : (j + 1) * b_c]
            )

    return out
