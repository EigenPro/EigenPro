import numpy as np
import torch
from .utils import (
    get_lens,
    get_ohe,
    hamming_dist,
)


def hamming_ker_exp(
    seqs_x: str | torch.Tensor,
    seqs_y: str | torch.Tensor = None,
    alphabet_name: str = "dna",
    bandwidth: float = 1,
    lag: int = 1,
):
    """ Exponential kernel based on Hamming distance.

    Args:
        seqs_x: input sequences of shape (n_sample, seq_len).
        seqs_y: input sequences of shape (n_sample, seq_len).
        alphabet_name: alphabet name.
        bandwidth: kernel bandwidth.
        lag: lag for computing Hamming distance.

    Returns:
        kernel matrix of shape (n_sample, n_sample).
    """
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    return torch.tensor(np.exp(-h_dists / bandwidth))


def hamming_ker_dot(seqs_x: str|torch.Tensor, seqs_y: str|torch.Tensor=None, alphabet_name: str="dna", lag: int=1):
    """ Dot product kernel based on Hamming distance.

    Args:
        seqs_x: input sequences of shape (n_sample, seq_len).
        seqs_y: input sequences of shape (n_sample, seq_len).
        alphabet_name: alphabet name.
        lag: lag for computing Hamming distance.

    Returns:
        kernel matrix of shape (n_sample, n_sample).
    """
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    x_lens = get_lens(get_ohe(seqs_x))
    y_lens = get_lens(get_ohe(seqs_y))
    max_len = np.max(
        [
            np.tile(x_lens[:, None], (1, len(y_lens))),
            np.tile(y_lens[None, :], (len(x_lens), 1)),
        ],
        axis=0,
    )
    dot = max_len - h_dists
    return torch.tensor(dot / np.sqrt(x_lens[:, None] * y_lens[None, :]))


def hamming_ker_imq(
    seqs_x: str|torch.Tensor, seqs_y: str|torch.Tensor=None, alphabet_name: str="dna", scale: float=1, beta: float=1 / 2, lag: int=1
):
    """ Inverse multi-quadratic kernel based on Hamming distance.

    Args:
        seqs_x: input sequences of shape (n_sample, seq_len).
        seqs_y: input sequences of shape (n_sample, seq_len).
        alphabet_name: alphabet name.
        scale: kernel scale.
        beta: kernel beta.
        lag: lag for computing Hamming distance.

    Returns:
        kernel matrix of shape (n_sample, n_sample).
    """
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    return torch.tensor((1 + scale) ** beta / (scale + h_dists) ** beta)
