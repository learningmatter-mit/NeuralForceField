"""Functions to deal with the GPU and the CUDA driver"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import nvidia_smi
import torch


def batch_to(batch: Dict[str, list | torch.Tensor], device: str) -> Dict[str, List | torch.Tensor]:
    """Send batch to device

    Args:
        batch (dict): batch of data
        device (str): device to send data to

    Returns:
        dict: batch of data on device
    """
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, "to") else val
    return gpu_batch


def detach(val: torch.Tensor, to_numpy: bool = False) -> torch.Tensor | np.ndarray:
    """Detach GPU tensor

    Args:
        val (tensor): tensor to detach
        to_numpy (bool, optional): convert to numpy. Defaults to False.

    Returns:
        tensor: detached tensor
    """
    if to_numpy:
        return val.detach().cpu().numpy() if hasattr(val, "detach") else val
    return val.detach().cpu() if hasattr(val, "detach") else val


def batch_detach(
    batch: Dict[str, List | torch.Tensor], to_numpy: bool = False
) -> Dict[str, List | torch.Tensor]:
    """Detach batch of GPU tensors

    Args:
        batch (dict): batches of data/prediction
        to_numpy (bool, optional): convert to numpy. Defaults to False.

    Returns:
        dict: detached batch
    """
    detach_batch = dict()
    for key, val in batch.items():
        if isinstance(val, list):
            detach_batch[key] = [detach(sub_val, to_numpy=to_numpy) for sub_val in val]
        else:
            detach_batch[key] = detach(val, to_numpy=to_numpy)

        if to_numpy:
            detach_batch[key] = np.array(detach_batch[key])

    return detach_batch


def to_cpu(
    batch: Dict[str, List | torch.Tensor],
) -> Dict[str, List | torch.Tensor]:
    """Send batch to CPU

    Args:
        batch (dict): batch of data

    Returns:
        dict: batch of data on CPU
    """
    cpu_batch = {}
    for key, val in batch.items():
        cpu_batch[key] = val.detach().cpu() if hasattr(val, "to") else val
    return cpu_batch


def cuda_devices_sorted_by_free_mem() -> list[int]:
    """List available CUDA devices sorted by increasing available memory.

    To get the device with the most free memory, use the last list item.

    Taken from: CHGNet (https://github.com/CederGroupHub/chgnet)

    Returns:
        list[int]: list of CUDA devices sorted by increasing available memory
    """
    free_memories = []
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        free_memories.append(info.free)
    nvidia_smi.nvmlShutdown()

    return sorted(range(len(free_memories)), key=lambda x: free_memories[x])


def get_final_device(device: str) -> str:
    """Get final device to use

    Args:
        device (str): device to use

    Returns:
        str: final device to use
    """
    if "cuda" in device and torch.cuda.is_available():
        try:
            return f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"
        except nvidia_smi.NVMLError:
            return "cuda:0"
        return f"cuda:{cuda_devices_sorted_by_free_mem()[-1]}"
    return "cpu"
