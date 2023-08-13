'''Utility functions for performing fast SVD.'''

import numpy as np
import scipy.linalg as linalg


def top_q_eig(matrix: np.ndarray, q: int) -> (np.ndarray, np.ndarray, float):
    """
    Find the top q eigenvalues and eigenvectors of a matrix using scipy.linalg.eigh.

    Args:
    - matrix (np.ndarray): Symmetric matrix of shape (n, n).
    - q (int): Number of top eigenvalues/eigenvectors to retrieve.

    Returns:
    - eigenvalues (np.ndarray): Array of shape (q,) containing top q eigenvalues.
    - eigenvectors (np.ndarray): Array of shape (n, q) containing corresponding eigenvectors.
    - next_eigenvalue (float): The (q+1)-th eigenvalue.
    """
    
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    assert matrix.shape[0] == matrix.shape[1], "Matrix should be square."
    
    n_sample = matrix.shape[0]
    eigenvalues, eigenvectors = linalg.eigh(matrix, subset_by_index=[n_sample - q - 1, n_sample - 1])
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    return eigenvalues[:q], eigenvectors[:, :q], eigenvalues[q]
