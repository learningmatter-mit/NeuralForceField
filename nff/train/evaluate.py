import copy
import torch
from tqdm import tqdm

from nff.utils.cuda import batch_to, batch_detach
from nff.data.dataset import concatenate_dict


def evaluate(model,
             loader,
             loss_fn,
             device,
             return_results=True,
             loss_is_normalized=True,
             submodel=None,
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

        use_device = copy.deepcopy(device)
        while True:
            batch = batch_to(batch, use_device)
            model.to(use_device)
            try:
                # e.g. if the result is a sum of results from two models,
                # and you just want the prediction of one of those models
                if submodel is not None:
                    results = getattr(model, submodel)(batch)
                else:
                    results = model(batch, **kwargs)
                results = batch_to(batch_detach(results), use_device)
                eval_batch_loss = loss_fn(batch, results).data.cpu().numpy()

                break

            except RuntimeError as err:
                if 'CUDA out of memory' in str(err):
                    print(("CUDA out of memory. Doing this batch "
                           "on cpu."))
                    use_device = "cpu"
                    torch.cuda.empty_cache()

                else:
                    raise err

        if loss_is_normalized:
            eval_loss += eval_batch_loss * vsize
        else:
            eval_loss += eval_batch_loss

        all_results.append(batch_detach(results))
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
