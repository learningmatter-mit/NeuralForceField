import torch
from nff.utils.scatter import compute_grad
from nff.nn.modules import ConfAttention
EPS = 1e-15


def update_boltz(conf_fp, weight, boltz_nn):
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
    Returns:
        boltzmann_fp (torch.Tensor): updated fingerprint
    """

    if boltz_nn is None:
        boltzmann_fp = conf_fp * weight
    # otherwise concatenate the weight with the fingerprint
    # and put it through the boltzmann nn
    else:
        weight_tens = torch.Tensor([weight]).to(conf_fp.device)
        new_fp = torch.cat((conf_fp, weight_tens))
        boltzmann_fp = boltz_nn(new_fp)
    return boltzmann_fp


def conf_pool(mol_size,
              boltzmann_weights,
              mol_fp_nn,
              boltz_nns,
              conf_fps,
              head_pool="concatenate"):
    """
    Pool atomic representations of conformers into molecular fingerprint,
    and then add those fingerprints together with Boltzmann weights.
    Args:
        mol_size (int): number of atoms per molecle
        boltzmann_weights (torch.Tensor): tensor of length num_conf
            with boltzmann weights of each fonroerm.
        mol_fp_nn (torch.nn.Module): network that converts the sum
            of atomic fingerprints into a molecular fingerprint.
        boltz_nns (list[torch.nn.Module]): nns that take a molecular
            fingerprint and boltzmann weight as input and returns
            a new fingerprint.
        conf_fps (torch.Tensor): fingerprints for each conformer
        head_pool (str): how to combine species feature vectors from
            the different `boltz_nns`.
    Returns:
        final_fp (torch.Tensor): H-dimensional tensor, where
            H is the number of features in the molecular fingerprint.
    """

    final_fps = []
    final_weights = []
    for boltz_nn in boltz_nns:
        # if boltz_nn is an instance of ConfAttention,
        # put all the conformers and their weights into
        # the attention pooler and return

        if isinstance(boltz_nn, ConfAttention):
            final_fp, learned_weights = boltz_nn(
                conf_fps=conf_fps,
                boltzmann_weights=boltzmann_weights)
        else:
            # otherwise get a new fingerprint for each conformer
            # based on its Boltzmann weight

            boltzmann_fps = []
            for i, conf_fp in enumerate(conf_fps):
                weight = boltzmann_weights[i]
                boltzmann_fp = update_boltz(
                    conf_fp=conf_fp,
                    weight=weight,
                    boltz_nn=boltz_nn)
                boltzmann_fps.append(boltzmann_fp)

            boltzmann_fps = torch.stack(boltzmann_fps)
            learned_weights = boltzmann_weights

            # sum all the conformer fingerprints
            final_fp = boltzmann_fps.sum(dim=0)

        final_fps.append(final_fp)
        final_weights.append(learned_weights)

    # combine the fingerprints produced by the different
    # `boltz_nns`
    if head_pool == "concatenate":
        final_fp = torch.cat(final_fps, dim=-1)
    elif head_pool == "sum":
        final_fp = torch.stack(final_fps).sum(dim=0)
    else:
        raise NotImplementedError

    final_weights = torch.stack(final_weights)
    return final_fp, final_weights


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
