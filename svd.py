'''Utility functions for performing fast SVD.'''

import scipy.linalg as linalg
import numpy as np


def top_q_eig(matrix, top_q):
    """
    Find the top q eigenvalues and eigenvectors of a matrix using torch.lobpcg.

    Args:
    - matrix (torch.Tensor): Symmetric matrix of shape (n, n).
    - q (int): Number of top eigenvalues/eigenvectors to retrieve.

    Returns:
    - eigenvalues (torch.Tensor): Tensor of shape (q,) containing top q eigenvalues.
    - eigenvectors (torch.Tensor): Tensor of shape (n, q) containing corresponding eigenvectors.
    """

    # Ensure the matrix is symmetric and in numpy.
    if not isinstance(matrix, np.ndarray):
        matrix =  np.array(matrix)
    assert matrix.shape[0] == matrix.shape[1], "Matrix should be square."

    n_sample = matrix.shape[0]

    # Find eigenvalues and eigenvectors.
    # eigenvalues, eigenvectors = torch.lobpcg(matrix,k=q+1)
    eigenvalues, eigenvectors = linalg.eigh( matrix,
                                             subset_by_index = [n_sample - top_q - 1, n_sample - 1],eigvals_only=False )
    eigenvalues  = eigenvalues[::-1]
    eigenvectors = eigenvectors[:,::-1]


    return eigenvalues[:top_q], eigenvectors[:,:top_q],eigenvalues[top_q]
