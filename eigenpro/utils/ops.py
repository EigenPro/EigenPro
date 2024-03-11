from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor
import torch
from eigenpro.utils.tensor import RowDistributedTensor, BroadcastTensor, ColumnDistributedTensor, SummableDistributedTensor


############# CPU OPS to mimic torch.comm.<methodname> ####################

def gather(distributed_tensor, destination=None):
    if isinstance(distributed_tensor, RowDistributedTensor):
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


def distributed_kernel_evaluation(kernel_fn, X: BroadcastTensor, centers: RowDistributedTensor):
    with ThreadPoolExecutor() as executor:
        out = [
                executor.submit(kernel_fn, x, z) for x, z in zip(X.parts, centers.parts)
            ]
        kmat = ColumnDistributedTensor([k.result() for k in out], base_device_idx=centers.base_device_idx)
        del out
    return kmat

def distributed_matrix_multiply(mat1: ColumnDistributedTensor, mat2: RowDistributedTensor):
    with ThreadPoolExecutor() as executor:
        out = [
                executor.submit(torch.matmul, m1, m2) for m1, m2 in zip(mat1.parts, mat2.parts)
            ]
        mat3 = SummableDistributedTensor([k.result() for k in out], base_device_idx=mat2.base_device_idx)
        del out
    return mat3

# def distributed_matrix_slicing(mat: DistributedTensor, idx: DistributedTensor, offsets: torch.Tensor):
#     with ThreadPoolExecutor() as executor:
#         out = [
#                 executor.submit(torch.index_select, m, 0, i-o) for m, i, o in zip(mat.parts, idx.parts, offsets)
#             ]
#         mat3 = DistributedTensor([k.result() for k in out], base_device_idx=mat.base_device_idx)
#         del out
#     return mat3