"""Functions to deal with the GPU and the CUDA driver
"""

import torch

def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch

def batch_detach(batch):
    """detach batch of GPU tensors
    
    Args:
        batch (dict): batches of data/preidction
    
    Returns:
        TYPE: dict
    """
    detach_batch = dict()
    for key, val in batch.items():
        detach_batch[key] = val.detach().cpu() if hasattr(val, 'detach') else val
    return detach_batch

def to_cpu(batch):
    cpu_batch = {}
    for key, val in batch.items():
        cpu_batch[key] = val.detach().cpu() if hasattr(val, 'to') else val
    return cpu_batch