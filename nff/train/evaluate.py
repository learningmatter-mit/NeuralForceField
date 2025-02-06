"""This module contains functions for evaluating the model on a dataset"""

from __future__ import annotations

import copy

import torch
from tqdm import tqdm

from nff.data.dataset import concatenate_dict
from nff.utils.cuda import batch_detach, batch_to


def shrink_batch(batch: dict) -> dict:
    """Exclude certain keys from the batch that take up a lot of memory

    Args:
        batch (dict): the batch to shrink

    Returns:
        dict: the shrunk batch
    """
    bad_keys = ["nbr_list", "kj_idx", "ji_idx", "angle_list"]
    new_batch = {key: val for key, val in batch.items() if key not in bad_keys}

    return new_batch


def get_results(
    batch: dict, model: torch.nn.Module, device: str, submodel: str | torch.nn.Module, loss_fn: callable, **kwargs
) -> tuple[dict, float]:
    """Get the results of a batch from the model

    Args:
        batch (dict): the batch to evaluate
        model (torch.nn.Module): the model to evaluate
        device (str): the device to use
        submodel (str | torch.nn.Module): the submodel to use
        loss_fn (callable): the loss function to use
        **kwargs: additional keyword arguments

    Returns:
        tuple[dict, float]: the results of the batch and the loss
    """
    batch = batch_to(batch, device)
    model.to(device)
    if submodel is not None:  # noqa: SIM108
        results = getattr(model, submodel)(batch)
    else:
        results = model(batch, **kwargs)
    results = batch_to(batch_detach(results), device)
    if "forces" in results and "energy_grad" not in results:
        results["energy_grad"] = (
            [-x for x in results["forces"]] if isinstance(results["forces"], list) else -results["forces"]
        )
    eval_batch_loss = loss_fn(batch, results).data.cpu().numpy()

    return results, eval_batch_loss


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: callable,
    device: str,
    return_results: bool = True,
    loss_is_normalized: bool = True,
    submodel: torch.nn.Module | None = None,
    trim_batch: bool = False,
    catch_oom: bool = True,
    **kwargs,
) -> tuple[dict, dict, float]:
    """Evaluate the current state of the model using a given dataloader

    Args:
        model (torch.nn.Module): the model to evaluate
        loader (torch.utils.data.DataLoader): the dataloader to use
        loss_fn (callable): the loss function to use
        device (str): the device to use
        return_results (bool): whether to return the results
        loss_is_normalized (bool): whether the loss is normalized
        submodel (torch.nn.Module | None): the submodel to use
        trim_batch (bool): whether to exclude certain keys from the batch
        catch_oom (bool): whether to catch out of memory errors
        **kwargs: additional keyword arguments

    Returns:
        tuple[dict, dict, float]: the results of the evaluation, the batch used for evaluation,
         and the evaluation loss
    """
    model.eval()
    model.to(device)

    eval_loss = 0.0
    n_eval = 0

    all_results = []
    all_batches = []

    for batch in tqdm(loader):
        vsize = batch["nxyz"].size(0)
        n_eval += vsize

        if catch_oom:
            use_device = copy.deepcopy(device)
            while True:
                try:
                    results, eval_batch_loss = get_results(
                        batch=batch,
                        model=model,
                        device=use_device,
                        submodel=submodel,
                        loss_fn=loss_fn,
                        **kwargs,
                    )

                    break
                except RuntimeError as err:
                    if "CUDA out of memory" in str(err):
                        print("CUDA out of memory. Doing this batch on cpu.")
                        use_device = "cpu"
                        torch.cuda.empty_cache()
                    else:
                        raise err
        else:
            results, eval_batch_loss = get_results(
                batch=batch,
                model=model,
                device=device,
                submodel=submodel,
                loss_fn=loss_fn,
                **kwargs,
            )

        if loss_is_normalized:
            eval_loss += eval_batch_loss * vsize
        else:
            eval_loss += eval_batch_loss

        all_results.append(batch_detach(results))

        if trim_batch:
            batch = shrink_batch(batch)
        all_batches.append(batch_detach(batch))

        del results
        del batch

    # weighted average over batches
    if loss_is_normalized:
        eval_loss /= n_eval

    if not return_results:
        return {}, {}, eval_loss

    # this step can be slow,
    all_results = concatenate_dict(*all_results)
    all_batches = concatenate_dict(*all_batches)
    print(f"Embedding in results: {'embedding' in all_results}")

    return all_results, all_batches, eval_loss
