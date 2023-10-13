from collections import defaultdict, deque
from typing import Optional, List

import numpy as np
import torch


def to_device(nested_tensors, device, **kwargs):
    new_nested_tensors = dict()
    for key, value in nested_tensors.items():
        new_nested_tensors.update(
            {key : value.to(device, **kwargs)}
        )
    return new_nested_tensors


def to_torch(npy_dict):
    torch_dict = {}
    for key, value in npy_dict.items():
        torch_dict[key] = torch.from_numpy(value.copy())
    return torch_dict


def to_numpy(torch_dict):
    npy_dict = {}
    for key, value in torch_dict.items():
        if value.device == 'cpu':
            npy_dict[key] = value.detach().numpy()
        else:
            npy_dict[key] = value.cpu().detach().numpy()
    return npy_dict


