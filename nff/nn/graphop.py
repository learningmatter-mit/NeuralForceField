
import torch
import torch.nn as nns
import torch.nn.functional as F
from nff.utils.scatter import compute_grad
from nff.nn.modules import BondEnergyModule, GraphDis
from nff.data import get_neighbor_list

# Pooling function to get graph property
def batch_and_sum(dict_input, N, predict_keys, xyz, prior=False, bond_adj=None):
    """Separate the outputs back into batches, pool the results, compute gradient of 
    scalar properties if "_grad" is in the key name

    TO DO: handle cases for atom-wise and edge-wise outputs  
    
    Args:
        dict_input (dict): Description
        N (list): number of batches
        predict_keys (list): Description
        xyz (tensor): xyz of the molecule
    
    Returns:
        dict: batched and pooled results 
    """

    # remove this after @dskoda fix it 

    results = dict()

    for key, val in dict_input.items():
        #split 
        if key in predict_keys and "_grad" not in key:
            batched_prop = list(torch.split(val, N))

            for batch_idx in range(len(N)):
                batched_prop[batch_idx] = torch.sum(batched_prop[batch_idx], dim=0)

            results[key] = torch.stack(batched_prop).cpu()

        if key + "_grad" in predict_keys: # indicates that this key requires a grad computation

            grad = compute_grad(inputs=xyz, output=results[key])
            results[key + "_grad"] = grad.cpu()


    return results


def get_atoms_inside_cell(r, N, pbc):
    """Summary
    
    Args:
        r (torch.float): atomic embeddings 
        N (torch.long): Description
        pbc (troch.long): atomic embeddings
    
    Returns:
        TYPE: Description
    """
    N = N.tolist()
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