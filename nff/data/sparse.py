import torch
import torch.sparse as sp


def sparsify_tensor(tensor):
    """Convert a torch.Tensor into a torch.sparse.FloatTensor

    Args:
        tensor (torch.Tensor)

    returns:
        sparse (torch.sparse.Tensor)
    """
    ij = tensor.nonzero(as_tuple=False)

    if len(ij) > 0:
        v = tensor[ij[:, 0], ij[:, 1]]
        return sp.FloatTensor(ij.t(), v, tensor.size())
    else:
        return 0


def sparsify_array(array):
    """Convert a np.array into a torch.sparse.FloatTensor

    Args:
        array (np.array)

    returns:
        sparse (torch.sparse.Tensor)
    """
    return sparsify_tensor(torch.FloatTensor(array))
