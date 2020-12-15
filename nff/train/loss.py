import torch
from nff.utils import constants as const

__all__ = ["build_mse_loss"]

EPS = 1e-15


def build_general_loss(loss_coef, operation, correspondence_keys=None):
    """
    Build a general  loss function.

    Args:
        loss_coef (dict): dictionary containing the weight coefficients
            for each property being predicted.
            Example: `loss_coef = {'energy': rho, 'force': 1}`
        operation (function): a function that acts on the prediction and
            the target to produce a result (e.g. square it, put it through
            cross-entropy, etc.)
        correspondence_keys (dict): a dictionary that links an output key to
            a different key in the dataset.
            Example: correspondence_keys = {"autopology_energy_grad": "energy_grad"}
            This tells us that if we see "autopology_energy_grad" show up in the
            loss coefficient, then the loss should be calculated between
            the network's output "autopology_energy_grad" and the data in the dataset
            given by "energy_grad". This is useful if we're only outputting one quantity,
            such as the energy gradient, but we want two different outputs (such as
            "energy_grad" and "autopology_energy_grad") to be compared to it.

    Returns:
        mean squared error loss function

    """

    correspondence_keys = {} if (
        correspondence_keys is None) else correspondence_keys

    def loss_fn(ground_truth, results):
        """Calculates the MSE between ground_truth and results.

        Args:
            ground_truth (dict): e.g. `{'energy': 2, 'force': [0, 0, 0]}`
            results (dict):  e.g. `{'energy': 4, 'force': [1, 2, 2]}`

        Returns:
            loss (torch.Tensor)
        """

        assert all([k in results.keys() for k in loss_coef.keys()])
        assert all([k in [*ground_truth.keys(), *correspondence_keys.keys()]
                    for k in loss_coef.keys()])

        loss = 0.0
        for key, coef in loss_coef.items():

            if key not in ground_truth.keys():
                ground_key = correspondence_keys[key]
            else:
                ground_key = key

            targ = ground_truth[ground_key]
            pred = results[key].view(targ.shape)

            # select only properties which are given
            valid_idx = torch.bitwise_not(torch.isnan(targ))
            targ = targ[valid_idx]
            pred = pred[valid_idx]

            if len(targ) != 0:
                diff = operation(targ=targ, pred=pred)
                err_sq = coef * torch.mean(diff)
                loss += err_sq

        return loss

    return loss_fn


def mse_operation(targ, pred):
    """
    Square the difference of target and predicted.
    Args:
        targ (torch.Tensor): target
        pred (torch.Tensor): prediction
    Returns:
        diff (torch.Tensor): difference squared
    """
    targ = targ.to(torch.float)
    diff = (targ - pred) ** 2
    return diff


def cross_entropy(targ, pred):
    """
    Take the cross-entropy between predicted and target.
    Args:
        targ (torch.Tensor): target
        pred (torch.Tensor): prediction
    Returns:
        diff (torch.Tensor): cross-entropy.
    """

    targ = targ.to(torch.float)
    fn = torch.nn.BCELoss(reduction='none')
    diff = fn(pred, targ)

    return diff


def cross_entropy_sum(targ, pred):

    loss_fn = BCELoss(reduction='sum')
    loss = loss_fn(pred, targ)

    return loss


def logits_cross_entropy(targ, pred):

    targ = targ.to(torch.float)
    fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    diff = fn(pred, targ)

    return diff


def zhu_p(gap, expec_gap):
    p = torch.exp(-gap ** 2 / (2 * expec_gap ** 2))
    return p


def zhu_p_grad(gap, gap_grad, expec_gap):
    p = torch.exp(-gap ** 2 / (2 * expec_gap ** 2))
    p_grad = - p * gap * gap_grad / (expec_gap ** 2)

    return p_grad


def batch_zhu_p(batch,
                upper_key,
                lower_key,
                expec_gap,
                gap_shape=None):

    gap = batch[upper_key] - batch[lower_key]

    if gap_shape is not None:
        gap = gap.view(gap_shape)
    p = zhu_p(gap=gap,
              expec_gap=expec_gap)

    return p


def batch_zhu_p_grad(batch,
                     upper_key,
                     lower_key,
                     expec_gap,
                     gap_shape=None,
                     gap_grad_shape=None):

    num_atoms = batch['num_atoms'].long()
    gap = batch[upper_key] - batch[lower_key]

    # give it the same shape as the forces
    gap = torch.cat([geom_gap.expand(n, 3)
                     for geom_gap, n in zip(gap, num_atoms)])

    gap_grad = batch[upper_key + "_grad"] - batch[lower_key + "_grad"]

    if gap_shape is not None:
        gap = gap.view(gap_shape)
    if gap_grad_shape is not None:
        gap_grad = gap_grad.view(gap_grad_shape)

    p_grad = zhu_p_grad(
        gap=gap,
        gap_grad=gap_grad,
        expec_gap=expec_gap)

    return p_grad, gap, gap_grad


def results_to_batch(results, ground_truth):
    results_batch = results.copy()

    for key, val in ground_truth.items():
        if key not in results_batch:
            results_batch[key] = val

    return results_batch


def build_mse_loss(loss_coef, correspondence_keys=None):
    """
    Build MSE loss from loss_coef.
    Args:
        loss_coef, correspondence_keys: see `build_general_loss`.
    Returns:
        loss_fn (function): loss function
    """

    loss_fn = build_general_loss(loss_coef=loss_coef,
                                 operation=mse_operation,
                                 correspondence_keys=correspondence_keys)
    return loss_fn


def build_cross_entropy_loss(loss_coef, correspondence_keys=None):
    """
    Build cross-entropy loss from loss_coef.
    Args:
        loss_coef, correspondence_keys: see `build_general_loss`.
    Returns:
        loss_fn (function): loss function
    """

    loss_fn = build_general_loss(loss_coef=loss_coef,
                                 operation=cross_entropy,
                                 correspondence_keys=correspondence_keys)
    return loss_fn


def build_logits_cross_entropy_loss(loss_coef, correspondence_keys=None):
    """
    Build logits cross-entropy loss from loss_coef.
    Args:
        loss_coef, correspondence_keys: see `build_general_loss`.
    Returns:
        loss_fn (function): loss function
    """

    loss_fn = build_general_loss(loss_coef=loss_coef,
                                 operation=logits_cross_entropy,
                                 correspondence_keys=correspondence_keys)
    return loss_fn


def build_zhu_loss(loss_dict):

    lower_key = loss_dict["params"]["lower_energy"]
    upper_key = loss_dict["params"]["upper_energy"]
    expec_gap = loss_dict["params"]["expected_gap"] * \
        const.AU_TO_KCAL["energy"]
    loss_key = loss_dict["params"].get("loss_type", "mse")

    if loss_key == "mse":
        loss_type = mse_operation
    elif loss_key == "cross_entropy":
        loss_type = cross_entropy_sum

    coef = loss_dict["coef"]

    def loss_fn(ground_truth, results, **kwargs):

        targ_gap = ground_truth[upper_key] - ground_truth[lower_key]

        targ_p = zhu_p(gap=targ_gap, expec_gap=expec_gap)

        pred_gap = (results[upper_key] - results[lower_key]
                    ).view(targ_gap.shape)
        pred_p = zhu_p(gap=pred_gap, expec_gap=expec_gap)

        valid_idx = torch.bitwise_not(torch.isnan(targ_p))
        targ_p = targ_p[valid_idx]
        pred_p = pred_p[valid_idx]

        diff = loss_type(targ_p, pred_p)
        err_sq = coef * torch.mean(diff)

        loss = err_sq

        return loss

    return loss_fn


def build_zhu_grad_loss(loss_dict):

    lower_key = loss_dict["params"]["lower_energy"]
    upper_key = loss_dict["params"]["upper_energy"]
    expec_gap = loss_dict["params"]["expected_gap"] * \
        const.AU_TO_KCAL["energy"]
    loss_key = loss_dict["params"].get("loss_type", "mse")

    if loss_key == "mse":
        loss_type = mse_operation
    elif loss_key == "cross_entropy":
        loss_type = cross_entropy_sum
    else:
        raise NotImplementedError

    coef = loss_dict["coef"]

    def loss_fn(ground_truth, results, **kwargs):

        targ_p_grad, targ_gap, targ_gap_grad = batch_zhu_p_grad(
            batch=ground_truth,
            expec_gap=expec_gap,
            upper_key=upper_key,
            lower_key=lower_key)

        gap_shape = targ_gap.shape
        gap_grad_shape = targ_gap_grad.shape
        results_batch = results_to_batch(results=results,
                                         ground_truth=ground_truth)

        pred_p_grad, _, _ = batch_zhu_p_grad(batch=results_batch,
                                             upper_key=upper_key,
                                             lower_key=lower_key,
                                             expec_gap=expec_gap,
                                             gap_shape=gap_shape,
                                             gap_grad_shape=gap_grad_shape)

        valid_idx = torch.bitwise_not(torch.isnan(targ_p_grad))
        targ_p_grad = targ_p_grad[valid_idx]
        pred_p_grad = pred_p_grad[valid_idx]

        diff = loss_type(targ_p_grad, pred_p_grad)
        err_sq = coef * torch.mean(diff)

        loss = err_sq

        return loss

    return loss_fn


def name_to_func(name):
    dic = {
        "mse": build_mse_loss,
        "cross_entropy": build_cross_entropy_loss,
        "logits_cross_entropy": build_logits_cross_entropy_loss,
        "zhu": build_zhu_loss,
        "zhu_grad": build_zhu_grad_loss
    }
    func = dic[name]
    return func


def get_all_losses(multi_loss_dict):
    loss_fns = []
    for key, loss_list in multi_loss_dict.items():
        build_fn = name_to_func(key)

        # for backwards compatability
        if key == "mse":
            loss_coef = {sub_dic["params"]["key"]:
                         sub_dic["coef"] for sub_dic in loss_list}
            loss_fn = build_fn(loss_coef)
            loss_fns.append(loss_fn)
            continue

        for loss_dict in loss_list:
            loss_fn = build_fn(loss_dict=loss_dict)
            loss_fns.append(loss_fn)

    return loss_fns


def build_multi_loss(multi_loss_dict):
    loss_fns = get_all_losses(multi_loss_dict=multi_loss_dict)

    def calc_loss(*args, **kwargs):
        loss = 0.0
        for loss_fn in loss_fns:
            loss += loss_fn(*args, **kwargs)
        return loss
    return calc_loss
