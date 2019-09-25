import numpy as np

import torch

from nff.utils.cuda import batch_to, to_cpu
from nff.utils.scatter import compute_grad
from nff.data.dataset import concatenate_dict

def evaluate(model, loader, loss_fn, device, loss_is_normalized=True):
    """Evaluate the current state of the model using a given dataloader
    """

    model.to(device)

    eval_loss = 0.0
    n_eval = 0

    all_results = []
    all_batches = []

    for batch in loader:
        # append batch_size

        batch = batch_to(batch, device)

        vsize = batch['nxyz'].size(0)
        n_eval += vsize

        results = model(batch)
        results["nxyz"] = batch["nxyz"]

        eval_batch_loss = loss_fn(batch, results).data.cpu().numpy()

        if loss_is_normalized:
            eval_loss += eval_batch_loss * vsize
        else:
            eval_loss += eval_batch_loss

        all_results.append(to_cpu(results))
        all_batches.append(to_cpu(batch))

    # weighted average over batches
    if loss_is_normalized:
        eval_loss /= n_eval

    all_results = concatenate_dict(*all_results)
    all_batches = concatenate_dict(*all_batches)

    return all_results, all_batches, eval_loss
