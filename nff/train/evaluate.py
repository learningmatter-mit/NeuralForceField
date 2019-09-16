import numpy as np

import torch

from nff.utils.scatter import compute_grad
from nff.data.dataset import concatenate_dict
import pdb

def evaluate(model, loader, loss_fn, device, loss_is_normalized=True):
    """Evaluate the current state of the model using a given dataloader
    """

    model.to(device)

    eval_loss = 0.0
    n_eval = 0

    all_results = []
    all_batches = []
    all_other_results = []
    # pdb.set_trace()
    for batch in loader:
        # append batch_size

        # pdb.set_trace()

        # vsize = batch['nxyz'].size(-1)
        vsize = batch['nxyz'].size(0)
        n_eval += vsize

        results, other_results = model(batch, other_results=True)
        results["nxyz"] = batch["nxyz"]
        other_results["nxyz"] = batch["nxyz"]

        eval_batch_loss = loss_fn(batch, results).data.cpu().numpy()

        if loss_is_normalized:
            eval_loss += eval_batch_loss * vsize
        else:
            eval_loss += eval_batch_loss

        all_results.append(results)
        all_batches.append(batch)
        all_other_results.append(other_results)

    # weighted average over batches
    if loss_is_normalized:
        eval_loss /= n_eval

    all_results = concatenate_dict(*all_results)
    all_other_results = concatenate_dict(*all_other_results)
    all_batches = concatenate_dict(*all_batches)

    return all_results, all_batches, eval_loss, all_other_results