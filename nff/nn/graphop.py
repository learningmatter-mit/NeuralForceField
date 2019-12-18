import torch
import torch.nn as nns
import torch.nn.functional as F
from nff.utils.scatter import compute_grad

EPS = 1e-15


def split_and_sum(tensor, N):
    """spliting a torch Tensor into a list of uneven sized tensors,
    and sum each tensor and stack 

    Example: 
        A = torch.rand(10, 10)
        N = [4,6]
        split_and_sum(A, N).shape # (2, 10) 
    
    Args:
        tensor (torch.Tensor): tensors to be split and summed
        N (list): list of number of atoms 
    
    Returns:
        torch.Tensor: stacked tensor of summed smaller tensor 
    """
    batched_prop = list(torch.split(tensor, N))

    for batch_idx in range(len(N)):
        batched_prop[batch_idx] = torch.sum(batched_prop[batch_idx], dim=0)

    return torch.stack(batched_prop)

def batch_and_sum(dict_input, N, predict_keys, xyz):
    """Pooling function to get graph property.
        Separate the outputs back into batches, pool the results,
        compute gradient of scalar properties if "_grad" is in the key name.
    
    Args:
        dict_input (dict): Description
        N (list): number of batches
        predict_keys (list): Description
        xyz (tensor): xyz of the molecule
    
    Returns:
        dict: batched and pooled results 
    """

    results = dict()

    for key, val in dict_input.items():
        #split 
        if key in predict_keys and key + "_grad" not in predict_keys:
            results[key] = split_and_sum(val, N)
        elif key in predict_keys and key + "_grad" in predict_keys:
            results[key] = split_and_sum(val, N)
            grad = compute_grad(inputs=xyz, output=results[key])
            results[key + "_grad"] = grad
        # For the case only predicting gradient 
        elif key not in predict_keys and key + "_grad" in predict_keys:
            results[key] = split_and_sum(val, N)
            grad = compute_grad(inputs=xyz, output=results[key])
            results[key + "_grad"] = grad

    return results


def get_atoms_inside_cell(r, N, pbc):
    """Removes atoms outside of the unit cell which are carried in `r`
        to ensure correct periodic boundary conditions. Does that by discarding
        all atoms beyond N which are not in the reindexing mapping `pbc`.
    
    Args:
        r (torch.float): atomic embeddings 
        N (torch.long): number of atoms inside each graph
        pbc (troch.long): atomic embeddings
    
    Returns:
        torch.float: atomnic embedding tensors inside the cell 
    """
    N = N.to(torch.long).tolist()

    # make N a list if it is a int 
    if type(N) == int:
        N = [N]

    # selecting only the atoms inside the unit cell
    atoms_in_cell = [
        set(x.cpu().data.numpy())
        for x in torch.split(pbc, N)
    ]

    N = [len(n) for n in atoms_in_cell]

    atoms_in_cell = torch.cat([
        torch.LongTensor(list(x))
        for x in atoms_in_cell
    ])

    r = r[atoms_in_cell]

    return r, N


