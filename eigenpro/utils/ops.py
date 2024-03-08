from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor
import torch
from eigenpro.utils.tensor import DistributedTensor


############# CPU OPS to mimic torch.comm.<methodname> ####################

def gather(distributed_tensor, destination=None):
    if isinstance(distributed_tensor, DistributedTensor):
        return torch.cat(distributed_tensor.parts)
    else:
        return distributed_tensor

def scatter(tensor, chunk_sizes, devices=None):
    return torch.split(tensor, split_size_or_sections=chunk_sizes.tolist())

def broadcast(tensor, devices):
    return [tensor for _ in devices]

def reduce_add(distributed_tensor, destination=None):
    return sum(distributed_tensor.parts)

##########################################################################


def distributed_kernel_evaluation(kernel_fn, X: DistributedTensor, centers: DistributedTensor):
    with ThreadPoolExecutor() as executor:
        out = [
                executor.submit(kernel_fn, x, z) for x, z in zip(X.parts, centers.parts)
            ]
        kmat = DistributedTensor([k.result() for k in out])
        del out
    return kmat

def distributed_matrix_multiply(mat1: DistributedTensor, mat2: DistributedTensor):
    with ThreadPoolExecutor() as executor:
        out = [
                executor.submit(torch.matmul, m1, m2) for m1, m2 in zip(mat1.parts, mat2.parts)
            ]
        mat3 = DistributedTensor([k.result() for k in out])
        del out
    return mat3

def distributed_matrix_slicing(mat: DistributedTensor, idx: DistributedTensor, offsets: torch.Tensor):
    with ThreadPoolExecutor() as executor:
        out = [
                executor.submit(torch.index_select, m, 0, i-o) for m, i, o in zip(mat.parts, idx.parts, offsets)
            ]
        mat3 = DistributedTensor([k.result() for k in out])
        del out
    return mat3