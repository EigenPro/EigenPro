import torch


def hamming_imq(
    x1: torch.Tensor,
    x2: torch.Tensor,
    vocab_size: int,
    diag: bool = False,
    alpha: float = 1.0,
    beta: float = 0.5,
    batch_shape=torch.Size([]),
) -> torch.Tensor:
    """Compute the Hamming IMQ kernel between two sequences.

    Args:
        x1 : First sequence.
        x2 : Second sequence.
        vocab_size : Size of the vocabulary.
        diag : Compute the diagonal of the kernel matrix. Defaults to False.
        alpha : Alpha parameter of the IMQ kernel. Defaults to 1.0.
        beta : Beta parameter of the IMQ kernel. Defaults to 0.5.
        batch_shape : Set this if you want a separate kernel hyperparameters for each batch of input
        data.

    Returns:
        IMQ kernel between the two sequences.
    """
    # Unflatten the one-hot encoding
    x1 = x1.view(*x1.shape[:-1], -1, vocab_size)
    x2 = x2.view(*x2.shape[:-1], -1, vocab_size)

    x1_eq_x2 = torch.equal(x1, x2)

    if diag:
        if x1_eq_x2:
            res = ((1 + alpha) / alpha).pow(beta)
            skip_dims = [-1] * len(batch_shape)
            return res.expand(*skip_dims, x1.size(-3))
        else:
            dist = x1.size(-2) - (x1 * x2).sum(dim=(-1, -2))
            return imq(dist, alpha, beta)

    else:
        dist = hamming_dist(x1, x2, x1_eq_x2)

    return imq(dist, alpha, beta)


def hamming_dist(x1: torch.Tensor, x2: torch.Tensor, x1_eq_x2: bool) -> torch.Tensor:
    """Calculate the Hamming distance between two sequences.

    Args:
        x1 : First sequence.
        x2 : Second sequence.
        x1_eq_x2 : Are the two sequences the same?

    Returns:
        Hamming distance between the two sequences.
    """
    res = x1.size(-2) - (x1.unsqueeze(-3) * x2.unsqueeze(-4)).sum(dim=(-1, -2))
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)
    # Zero out negative values
    return res.clamp_min_(0)


def imq(dist: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Compute the Inverse Multi-Quadratic kernel.

    Args:
        dist : Distance between the two sequences.
        alpha : Alpha parameter of the IMQ kernel.
        beta : Beta parameter of the IMQ kernel.

    Returns:
        IMQ kernel between the two sequences.
    """
    return ((1 + alpha) / (alpha + dist)).pow(beta)
