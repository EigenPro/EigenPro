"""Implementation of kernel functions.
"""

import torch

EPSILON = 1e-12

def euclidean(samples: torch.Tensor, centers: torch.Tensor,
              squared: bool = True) -> torch.Tensor:
    '''Calculate the pointwise distance.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        squared: boolean.

    Returns:
        pointwise distances (n_sample, n_center).
    '''
    samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = torch.sum(centers**2, dim=1, keepdim=True)
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(torch.t(centers))
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)
    if not squared:
        distances.clamp_(min=0)        
        distances.sqrt_()

    return distances


def gaussian(samples: torch.Tensor, centers: torch.Tensor,
             bandwidth: float) -> torch.Tensor:
    '''Gaussian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def laplacian(samples: torch.Tensor, centers: torch.Tensor,
              bandwidth: float) -> torch.Tensor:
    '''Laplacian kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def dispersal(samples: torch.Tensor, centers: torch.Tensor,
              bandwidth: float, gamma: float) -> torch.Tensor:
    '''Dispersal kernel.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.
        gamma: dispersal factor.

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers)
    kernel_mat.pow_(gamma / 2.)
    kernel_mat.mul_(-1. / bandwidth)
    kernel_mat.exp_()
    return kernel_mat


def ntk_relu(X: torch.Tensor, Z: torch.Tensor,
             depth: int = 1, bias: float = 0) -> torch.Tensor:
    """NTK Relu Kernel.

    Args:
        depth: number of layers of the network
        bias: (default=0.)

    Returns: 
        The evaluation of nngp and ntk kernels for fully connected neural
        networks with ReLU nonlinearity.
    """
    from torch import acos, pi
    kappa_0 = lambda u: (1-acos(u)/pi)
    kappa_1 = lambda u: u*kappa_0(u) + (1-u.pow(2)).sqrt()/pi
    Z = Z if Z is not None else X
    norm_x = X.norm(dim=-1)[:, None].clip(min=EPSILON)
    norm_z = Z.norm(dim=-1)[None, :].clip(min=EPSILON)
    S = X @ Z.T
    N = S + bias**2
    for k in range(1, depth):
        in_ = (S/norm_x/norm_z).clip(min=-1+EPSILON,max=1-EPSILON)
        S = norm_x*norm_z*kappa_1(in_)
        N = N * kappa_0(in_) + S + bias**2
    return N

def ntk_relu_unit_sphere(X: torch.Tensor, Z: torch.Tensor,
                         depth: int = 1, bias: float = 0) -> torch.Tensor:
    """Computes NTK and NNGP kernels for a fully-connected ReLU network.

    This function calculates the Neural Tangent Kernel (NTK) and Neural
    Network Gaussian Process (NNGP) kernels for a fully-connected neural
    network with ReLU non-linearity. The inputs are assumed to be normalized
    to unit norm.

    Args:
        X (torch.Tensor): The first input tensor.
        Z (torch.Tensor): The second input tensor. If None, it will be set to
            X.
        depth (int, optional): The number of layers in the network. Defaults to
            1.
        bias (float, optional): The bias term to be added. Defaults to 0.

    Returns:
        torch.Tensor: The calculated NTK and NNGP kernels.
    """
    from torch import acos, pi
    kappa_0 = lambda u: (1-acos(u)/pi)
    kappa_1 = lambda u: u*kappa_0(u) + (1-u.pow(2)).sqrt()/pi
    Z = Z if Z is not None else X
    S = X @ Z.T
    N = S + bias**2
    for k in range(1, depth):
        in_ = (S).clip(min=-1+EPSILON,max=1-EPSILON)
        S = kappa_1(in_)
        N = N * kappa_0(in_) + S + bias**2
    return N
