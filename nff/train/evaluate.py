import numpy as np

import torch

from nff.utils.scatter import compute_grad


def evaluate(model, loader, loss_fn, device, loss_is_normalized=True):
    """Evaluate the current state of the model using a given dataloader
    """

    model.to(device)
    loader.to(device)

    eval_loss = 0.0
    n_eval = 0

    all_results = {
        'energy': [],
        'force': []
    }

    all_targets = {
        'energy': [],
        'force': []
    }

    for eval_batch in loader:
        xyz, a, bond_adj, bond_len, r, f, u, N = eval_batch
        xyz.requires_grad = True

        ground_truth = {
            'energy': u,
            'force': f
        }

        # append batch_size
        vsize = xyz.size(0)
        n_eval += vsize

        energy_nff = model(
            r=r,
            bond_adj=bond_adj,
            bond_len=bond_len,
            xyz=xyz,
            a=a,
            N=N
        )

        force_nff = -compute_grad(inputs=xyz, output=energy_nff)

        results = {
            'energy': energy_nff,
            'force': force_nff
        }

        eval_batch_loss = (
            loss_fn(ground_truth, results).data.cpu().numpy()
        )

        if loss_is_normalized:
            eval_loss += eval_batch_loss * vsize
        else:
            eval_loss += eval_batch_loss

        for key in all_results.keys():
            all_targets[key] += [ground_truth[key].cpu().reshape(-1).data.numpy()]
            all_results[key] += [results[key].cpu().reshape(-1).data.numpy()]

    # weighted average over batches
    if loss_is_normalized:
        eval_loss /= n_eval

    for dict_ in [all_results, all_targets]:
        for key, val in dict_.items():
            dict_[key] = np.concatenate(val, axis=0)

    return all_results, all_targets, eval_loss
