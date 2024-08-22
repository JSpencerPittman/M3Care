from torch import nn
import numpy as np
import copy
import torch
from typing import Optional


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


def pad_axis(arr: np.array, pad_to: int, axis: int) -> np.ndarray:
    """
    Pad a numpy array to the pad_to value on the specified axis.
    If the pad_to value is less than the size of the array, the data will be cutoff.

    Args:
        arr (np.array): Array to be padded.
        pad_to (int): Pad to value.
        axis (int): Axis to fill to on.

    Returns:
        np.ndarray: Array padded on specified axis.
    """

    padded_shape = tuple((pad_to if arr_axis == axis else dim)
                         for arr_axis, dim in enumerate(arr.shape))
    padded_arr = np.zeros_like(arr, shape=padded_shape)
    slices = tuple(slice(0, dim, 1) for dim in arr.shape)
    padded_arr[slices] = arr
    return padded_arr


def pad_axes(arr: np.ndarray, pad_to: tuple[int]) -> np.ndarray:
    """
    Pad a numpy array to the pad_to values.
    If the pad_to value is less than the size of the array, the data will be cutoff.

    Args:
        arr (np.array): Array to be padded.
        pad_to (int): Dimensions to pad the array to.

    Returns:
        np.ndarray: Array padded to match pad_to shape.
    """

    pad_to = tuple((dim if dim == -1 else pad_dim)
                   for dim, pad_dim in zip(arr.shape, pad_to.shape))
    padded_arr = np.zeros_like(arr, shape=pad_to)
    slices = tuple(slice(0, dim, 1) for dim in arr.shape)
    padded_arr[slices] = arr
    return padded_arr


def padded_stack(*arrs: np.ndarray, pad_to: Optional[tuple[int]] = None) -> np.ndarray:
    """
    Stack two numpy arrays. If the numpy arrays are inhomogenous then they will be
    padded in each dimension until they match in shape. If a custom shape is wanted
    then the pad_to argument can be used to enforce it, if any dimension in the pad_to
    shape is smaller than one of the proviced arrays along that same dimension then the
    array's data will be cutoff and not returned in the padded array. The pad_to shape
    defines each arrays shape before the stack operation so the final shape would be
    (number of arrays, *pad_to).

    Args:
        pad_to (Optional[tuple[int]], optional): Shape each subarray will be padded to.
        Defaults to None.

    Returns:
        np.ndarray: The stack of all passed in arrays.
    """

    baseline = arrs[0]
    assert all([arr.ndim == baseline.ndim for arr in arrs[1:]])

    new_dims = np.max(np.array([arr.shape for arr in arrs]), axis=0)
    if pad_to is not None:
        assert len(pad_to) == baseline.ndim
        new_dims = tuple((new_dim if pad_dim == -1 else pad_dim)
                         for new_dim, pad_dim in zip(new_dims, pad_to))

    new_dims = (len(arrs), *new_dims)
    padded_arr = np.zeros_like(baseline, shape=new_dims)
    for aidx, arr in enumerate(arrs):
        slices = tuple([aidx] + [slice(0, dim, 1) for dim in arr.shape])
        padded_arr[slices] = arr

    return padded_arr


def pad_missing(mat: np.array, mask: np.array):
    new_shape = mask.shape[0:1] + mat.shape[1:]
    full = np.zeros(new_shape)

    mask_idx = 0
    for full_idx, exists in enumerate(mask):
        if exists:
            full[full_idx] = mat[mask_idx]
            mask_idx += 1

    return full
