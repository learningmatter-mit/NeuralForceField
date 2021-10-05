import copy
import torch
from tqdm import tqdm

from nff.utils.cuda import batch_to, batch_detach
from nff.data.dataset import concatenate_dict


def shrink_batch(batch):
    """
    Exclude certain keys from the batch that take up a lot of memory
    """

    bad_keys = ['nbr_list', 'kj_idx', 'ji_idx',
                'angle_list']
    new_batch = {key: val for key, val in batch.items()
                 if key not in bad_keys}

    return new_batch


def get_results(batch,
                model,
                device,
                submodel,
                loss_fn,
                **kwargs):

    batch = batch_to(batch, device)
    model.to(device)
    if submodel is not None:
        results = getattr(model, submodel)(batch)
    else:
        results = model(batch, **kwargs)
    results = batch_to(batch_detach(results), device)
    eval_batch_loss = loss_fn(batch, results).data.cpu().numpy()

    return results, eval_batch_loss


def evaluate(model,
             loader,
             loss_fn,
             device,
             return_results=True,
             loss_is_normalized=True,
             submodel=None,
             trim_batch=False,
             catch_oom=True,
             **kwargs):
    """Evaluate the current state of the model using a given dataloader
    """

    model.eval()
    model.to(device)

    eval_loss = 0.0
    n_eval = 0

    all_results = []
    all_batches = []

    for batch in tqdm(loader):

        vsize = batch['nxyz'].size(0)
        n_eval += vsize

        if catch_oom:
            use_device = copy.deepcopy(device)
            while True:
                try:
                    results, eval_batch_loss = get_results(batch=batch,
                                                           model=model,
                                                           device=use_device,
                                                           submodel=submodel,
                                                           loss_fn=loss_fn,
                                                           **kwargs)

                    break
                except RuntimeError as err:
                    if 'CUDA out of memory' in str(err):
                        print(("CUDA out of memory. Doing this batch "
                               "on cpu."))
                        use_device = "cpu"
                        torch.cuda.empty_cache()
                    else:
                        raise err
        else:
            results, eval_batch_loss = get_results(batch=batch,
                                                   model=model,
                                                   device=device,
                                                   submodel=submodel,
                                                   loss_fn=loss_fn,
                                                   **kwargs)

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

    else:
        # this step can be slow,
        all_results = concatenate_dict(*all_results)
        all_batches = concatenate_dict(*all_batches)

        return all_results, all_batches, eval_loss
