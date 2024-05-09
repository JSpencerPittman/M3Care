from torch import (nn, Tensor)
import numpy as np
import copy
import torch
from typing import List


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def euclidean_dist(x, y):
    b = x.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(b, b)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(b, b).t()
    dist = xx+yy-2*torch.mm(x, y.t())
    return dist


def guassian_kernel(source, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
    n = source.size(0)
    L2_distance = euclidean_dist(source, source)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n**2-n)

    if bandwidth < 1e-3:
        bandwidth = 1

    bandwidth /= kernel_mul ** (kernel_num//2)
    bandwidth_list = [bandwidth*(kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance/bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)/len(kernel_val)


def pad_axis(arr: np.array, fill_to: int, axis: int):
    assert axis < arr.ndim

    curr_dim = arr.shape[axis]
    padding = [(0, 0)] * arr.ndim
    padding[axis] = (0, fill_to-curr_dim)
    return np.pad(arr, padding)


def padded_stack(mat: List[np.array], fill_dims=None):
    ndim = mat[0].ndim

    if fill_dims is None:
        fill_dims = np.max([sub.shape for sub in mat], axis=0)
    elif ndim == 1 and type(fill_dims) == int:
        fill_dims = [fill_dims]
    else:
        max_dims = np.max([sub.shape for sub in mat], axis=0)
        fill_dims = [mdim if dim == -1 else dim for mdim,
                     dim in zip(max_dims, fill_dims)]

    padded_mats = []
    for submat in mat:
        padding = [(0, fill_to-dim)
                   for dim, fill_to in zip(submat.shape, fill_dims)]
        padded_mats.append(np.pad(submat, padding))

    return np.array(padded_mats)


def pad_missing(mat: np.array, mask: np.array):
    new_shape = mask.shape[0:1] + mat.shape[1:]
    full = np.zeros(new_shape)

    mask_idx = 0
    for full_idx, exists in enumerate(mask):
        if exists:
            full[full_idx] = mat[mask_idx]
            mask_idx += 1

    return full
