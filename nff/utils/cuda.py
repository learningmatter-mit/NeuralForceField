"""Functions to deal with the GPU and the CUDA driver
"""

import nvidia_smi


def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch


def detach(val):
    return val.detach().cpu() if hasattr(val, "detach") else val


def batch_detach(batch):
    """detach batch of GPU tensors

    Args:
        batch (dict): batches of data/preidction

    Returns:
        TYPE: dict
    """
    detach_batch = dict()
    for key, val in batch.items():
        if type(val) is list:
            detach_batch[key] = [detach(sub_val) for sub_val in val]
        else:
            detach_batch[key] = detach(val)
    return detach_batch


def to_cpu(batch):
    cpu_batch = {}
    for key, val in batch.items():
        cpu_batch[key] = val.detach().cpu() if hasattr(val, 'to') else val
    return cpu_batch


def cuda_devices_sorted_by_free_mem() -> list[int]:
    """List available CUDA devices sorted by increasing available memory.

    To get the device with the most free memory, use the last list item.

    Taken from: CHGNet (https://github.com/CederGroupHub/chgnet)
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
