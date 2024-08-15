"""Utility functions for performing fast SVD."""

import torch
import numpy as np
import scipy as sp


class EigenSystem:
    """Represents an eigensystem with eigenvalues and eigenvectors.

    The number of eigenvalues and eigenvectors come primarily in pairs.
    With `n` such pairs, we can have `n` or `n + 1` eigenvalues.

    Attributes:
        _values (np.ndarray): Array of eigenvalues in descending order.
        _vectors (np.ndarray): Array of eigenvectors corresponding to the
            eigenvalues.
        _num (int): Number of pairs of eigenvalues and eigenvectors.
    """

    def __init__(self, values: np.ndarray, vectors: np.ndarray) -> None:
        """Initializes the EigenSystem with given eigenvalues and eigenvectors.

        Args:
            values (np.ndarray): Array of eigenvalues.
            vectors (np.ndarray): Array of eigenvectors.

        Raises:
            AssertionError: If the number of eigenvalues is less than the
                number of eigenvectors.
        """
        assert len(values) == vectors.shape[1] or len(values) == 1 + vectors.shape[1]
        self._values = values
        self._vectors = vectors
        self._num = min(len(values), len(vectors))

    @property
    def min_value(self) -> float:
        """Gets the smallest eigenvalue of the eigensystem.

        Returns:
            float: The smallest eigenvalue.
        """
        return self._values[-1]

    @property
    def size(self) -> int:
        """Gets the size of the eigensystem.

        Returns:
            int: The number of eigenvalues and eigenvectors.
        """
        return len(self._values) - 1

    @property
    def values(self) -> np.ndarray:
        """Gets the eigenvalues in descending order.

        Returns:
            np.ndarray: Array of eigenvalues.
        """
        return self._values[:-1]

    @property
    def vectors(self) -> np.ndarray:
        """Gets the eigenvectors of the eigensystem.

        Returns:
            np.ndarray: Array of eigenvectors.
        """
        return self._vectors[:, : self.size]


def top_q_eig(matrix: torch.Tensor, q: int) -> EigenSystem:
    """Finds the top `q` eigenvalues and eigenvectors of matrix.

    This function returns the top `q + 1` eigenvalues but only the top `q`
    eigenvectors.

    Args:
        matrix (torch.Tensor): Symmetric matrix of shape (n, n).
        q (int): Number of top eigenvalues/eigenvectors to retrieve.

    Returns:
        EigenSystem: EigenSystem object with top `q + 1` eigenvalues and top
            `q` corresponding eigenvectors.

    Raises:
        AssertionError: If the matrix is not square.
    """
    device = matrix.device
    matrix = matrix.cpu().data.numpy()
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    assert matrix.shape[0] == matrix.shape[1], "Matrix should be square."

    n_sample = matrix.shape[0]
    eigenvalues, eigenvectors = sp.linalg.eigh(
        matrix, subset_by_index=(n_sample - q - 1, n_sample - 1)
    )
    eigenvalues = torch.from_numpy(np.flip(eigenvalues).copy()).to(device)
    eigenvectors = torch.from_numpy(np.fliplr(eigenvectors).copy()).to(device)

    return EigenSystem(eigenvalues, eigenvectors)
