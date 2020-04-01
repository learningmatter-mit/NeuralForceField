import torch
from nff.utils.scatter import compute_grad

EPS = 1e-15


def update_boltz(conf_fp, weight, boltz_nn, extra_feats=None):
    """
    Given a conformer fingerprint and Boltzmann weight,
    return a new updated fingerprint.
    Args:
        conf_fp (torch.Tensor): molecular finerprint of
            a conformer
        weight (float): Boltzmann weight
        boltz_nn (torch.nn.Module): network that converts
            the fingerprint and weight into a new
            fingerprint. If None, just multiply the Boltzmann
            factor with the fingerprint.
        extra_feats (torch.Tensor): extra fingerprint that
            gets tacked on to the end of the SchNet-generated
            fingerprint.
    Returns:
        boltzmann_fp (torch.Tensor): updated fingerprint
    """
    # if no boltzmann nn, just multiply
    if extra_feats is not None:
        extra_feats = extra_feats.to(conf_fp.device)
        conf_fp = torch.cat((conf_fp, extra_feats))

    if boltz_nn is None:
        boltzmann_fp = conf_fp * weight
    # otherwise concatenate the weight with the fingerprint
    # and put it through the boltzmann nn
    else:
        weight_tens = torch.Tensor([weight]).to(conf_fp.device)
        new_fp = torch.cat((conf_fp, weight_tens))
        boltzmann_fp = boltz_nn(new_fp)
    return boltzmann_fp


def conf_pool(smiles_fp, mol_size, boltzmann_weights, mol_fp_nn, boltz_nn, extra_feats=None):
    """
    Pool atomic representations of conformers into molecular fingerprint,
    and then add those fingerprints together with Boltzmann weights.
    Args:
        smiles_fp (torch.Tensor): (mol_size * num_conf) x M fingerprint,
            where mol_size is the number of atoms per molecle, num_conf
            is the number of conformers for the species, and M is the
            number of features in the fingerprint.
        mol_size (int): number of atoms per molecle
        boltzmann_weights (torch.Tensor): tensor of length num_conf
            with boltzmann weights of each fonroerm.
        mol_fp_nn (torch.nn.Module): network that converts the sum
            of atomic fingerprints into a molecular fingerprint.
        boltz_nn (torch.nn.Module): nn that takes a molecular
            fingerprint and boltzmann weight as input and returns
            a new fingerprint. 
        extra_feats (torch.Tensor): extra fingerprint that
            gets tacked on to the end of the SchNet-generated
            fingerprint.
    Returns:
        final_fp (torch.Tensor): H-dimensional tensor, where
            H is the number of features in the molecular fingerprint.
    """

    # total number of atoms
    num_atoms = smiles_fp.shape[0]
    # unmber of conformers
    num_confs = num_atoms // mol_size
    N = [mol_size] * num_confs
    conf_fps = []

    # split the atomic fingerprints up by conformer
    for atomic_fps in torch.split(smiles_fp, N):
        # sum them an then convert to molecular fp
        summed_atomic_fps = atomic_fps.sum(dim=0)
        mol_fp = mol_fp_nn(summed_atomic_fps)
        # add to the list of conformer fp's
        conf_fps.append(mol_fp)

    # get a new fingerprint for each conformer based on its Boltzmann weight
    boltzmann_fps = []
    for i, conf_fp in enumerate(conf_fps):
        weight = boltzmann_weights[i]
        boltzmann_fp = update_boltz(conf_fp=conf_fp,
                                    weight=weight,
                                    boltz_nn=boltz_nn,
                                    extra_feats=extra_feats)
        boltzmann_fps.append(boltzmann_fp)

    boltzmann_fps = torch.stack(boltzmann_fps)

    # sum all the conformer fingerprints
    final_fp = boltzmann_fps.sum(dim=0)

    return final_fp


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
        # split
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
