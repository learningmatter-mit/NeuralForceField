"""Functions to deal with the GPU and the CUDA driver
"""

import torch

def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch


def to_cpu(batch):
    cpu_batch = {}
    for key, val in batch.items():
        cpu_batch[key] = val.detach().cpu() if hasattr(val, 'to') else val
    return cpu_batch
