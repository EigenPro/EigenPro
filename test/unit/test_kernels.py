from eigenpro.kernels import (
    euclidean,
    gaussian,
    laplacian,
    dispersal,
    ntk_relu,
    ntk_relu_unit_sphere,
    hamming_imq,
)
import torch
import pytest


@pytest.fixture
def x():
    return torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float32)


@pytest.fixture
def z():
    return torch.tensor([[7, 8, 9, 10, 11, 12]], dtype=torch.float32)


@pytest.fixture
def x_1hot():
    return torch.tensor([[1, 0, 0, 1, 0, 0]], dtype=torch.float32)


@pytest.fixture
def z_1hot():
    return torch.tensor([[0, 1, 0, 0, 1, 0]], dtype=torch.float32)


@pytest.mark.parametrize("squared", [(True), (False)])
def test_euclidean(x, z, squared):
    euclidean(x, z, squared=squared)


@pytest.mark.parametrize("bandwidth", [1.0, 2.0])
def test_gaussian(x, z, bandwidth):
    gaussian(x, z, bandwidth)


@pytest.mark.parametrize("bandwidth", [1.0, 2.0])
def test_laplacian(x, z, bandwidth):
    laplacian(x, z, bandwidth)


@pytest.mark.parametrize(
    "bandwidth, gamma", [(1.0, 1.0), (2.0, 2.0), (1.0, 2.0), (2.0, 1.0)]
)
def test_dispersal(x, z, bandwidth, gamma):
    dispersal(x, z, bandwidth, gamma)


@pytest.mark.parametrize(
    "depth, bias", [(1, 0.0), (2, 0.0), (1, 1.0), (2, 1.0), (1, 2.0), (2, 2.0)]
)
def test_ntk_relu(x, z, depth, bias):
    ntk_relu(x, z, depth, bias)


@pytest.mark.parametrize(
    "depth, bias", [(1, 0.0), (2, 0.0), (1, 1.0), (2, 1.0), (1, 2.0), (2, 2.0)]
)
def test_ntk_relu_unit_sphere(x, z, depth, bias):
    ntk_relu_unit_sphere(x, z, depth, bias)


@pytest.mark.parametrize(
    "vocab_size, batch_shape",
    [(3, torch.Size([1])), (2, torch.Size([2]))],
)
def test_hamming_imq(x_1hot, z_1hot, vocab_size, batch_shape):
    hamming_imq(x_1hot, z_1hot, vocab_size, batch_shape)
