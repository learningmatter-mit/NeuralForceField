import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from nff.utils import constants as const

__all__ = ["build_mse_loss"]

EPS = 1e-15


def build_general_loss(loss_coef,
                       operation,
                       correspondence_keys=None,
                       cutoff=None):
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
    cutoff = {} if (cutoff is None) else cutoff

    def loss_fn(ground_truth,
                results):
        """Calculates the MSE between ground_truth and results.

        Args:
            ground_truth (dict): e.g. `{'energy': 2, 'force': [0, 0, 0]}`
            results (dict):  e.g. `{'energy': 4, 'force': [1, 2, 2]}`

        Returns:
            loss (torch.Tensor)
        """

        # assert all([k in results.keys() for k in loss_coef.keys()])
        # assert all([k in [*ground_truth.keys(), *correspondence_keys.keys()]
        #             for k in loss_coef.keys()])

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
            if key in cutoff:
                valid_idx *= (targ <= cutoff[key])

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


def rms_operation(targ, pred):
    return mse_operation(targ, pred) ** 0.5


def mae_operation(targ, pred):
    targ = targ.to(torch.float)
    diff = abs(targ - pred)
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


def cross_entropy_sum(targ,
                      pred,
                      pos_weight=None):

    if pos_weight is None:
        loss_fn = torch.nn.BCELoss(reduction='sum')
        loss = loss_fn(pred, targ)
    else:
        loss = (-pos_weight * targ * torch.log(pred) -
                (1 - targ) * torch.log(1 - pred))

    return loss


def get_cross_entropy_loss(pos_weight):
    if pos_weight is None:
        def fn(targ, pred):
            loss_fn = torch.nn.BCELoss(reduction='sum')
            loss = loss_fn(pred, targ)
            return loss
    else:
        def fn(targ, pred):
            loss = (-pos_weight * targ * torch.log(pred) -
                    (1 - targ) * torch.log(1 - pred))
            return loss
    return fn


def logits_cross_entropy(targ, pred):

    targ = targ.to(torch.float)
    fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    diff = fn(pred, targ)

    return diff


def zhu_p(gap,
          expec_gap,
          func_type):

    if func_type == "gaussian":
        p = torch.exp(-gap ** 2 / (2 * expec_gap ** 2))
    elif func_type == "exponential":
        p = torch.exp(-gap / expec_gap)
    else:
        raise NotImplementedError

    return p


def zhu_p_grad(gap,
               gap_grad,
               expec_gap,
               func_type):

    if func_type == "gaussian":
        p = torch.exp(-gap ** 2 / (2 * expec_gap ** 2))
        p_grad = - p * gap * gap_grad / (expec_gap ** 2)
    elif func_type == "exponential":
        p = torch.exp(-gap / expec_gap)
        p_grad = - p * gap_grad / expec_gap

    else:
        raise NotImplementedError

    return p_grad


def batch_zhu_p(batch,
                upper_key,
                lower_key,
                expec_gap,
                func_type,
                gap_shape=None):

    gap = batch[upper_key] - batch[lower_key]

    if gap_shape is not None:
        gap = gap.view(gap_shape)
    p = zhu_p(gap=gap,
              expec_gap=expec_gap,
              func_type=func_type)

    return p


def batch_zhu_p_grad(batch,
                     upper_key,
                     lower_key,
                     expec_gap,
                     func_type,
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
        expec_gap=expec_gap,
        func_type=func_type)

    return p_grad, gap, gap_grad


def results_to_batch(results, ground_truth):
    results_batch = results.copy()

    for key, val in ground_truth.items():
        if key not in results_batch:
            results_batch[key] = val

    return results_batch


def build_mse_loss(loss_coef,
                   correspondence_keys=None,
                   cutoff=None):
    """
    Build MSE loss from loss_coef.
    Args:
        loss_coef, correspondence_keys: see `build_general_loss`.
    Returns:
        loss_fn (function): loss function
    """

    loss_fn = build_general_loss(loss_coef=loss_coef,
                                 operation=mse_operation,
                                 correspondence_keys=correspondence_keys,
                                 cutoff=cutoff)
    return loss_fn


def build_mae_loss(loss_coef,
                   correspondence_keys=None,
                   cutoff=None):
    """
    Build MSE loss from loss_coef.
    Args:
        loss_coef, correspondence_keys: see `build_general_loss`.
    Returns:
        loss_fn (function): loss function
    """

    loss_fn = build_general_loss(loss_coef=loss_coef,
                                 operation=mae_operation,
                                 correspondence_keys=correspondence_keys,
                                 cutoff=cutoff)
    return loss_fn


def build_rmse_loss(loss_coef,
                    correspondence_keys=None,
                    cutoff=None):
    """
    Build MSE loss from loss_coef.
    Args:
        loss_coef, correspondence_keys: see `build_general_loss`.
    Returns:
        loss_fn (function): loss function
    """

    loss_fn = build_general_loss(loss_coef=loss_coef,
                                 operation=rms_operation,
                                 correspondence_keys=correspondence_keys,
                                 cutoff=cutoff)
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


def get_p(ground_truth,
          results,
          upper_key,
          lower_key,
          expec_gap,
          func_type):

    targ_gap = ground_truth[upper_key] - ground_truth[lower_key]

    targ_p = zhu_p(gap=targ_gap,
                   expec_gap=expec_gap,
                   func_type=func_type)

    pred_gap = (results[upper_key] - results[lower_key]
                ).view(targ_gap.shape)
    pred_p = zhu_p(gap=pred_gap,
                   expec_gap=expec_gap,
                   func_type=func_type)

    valid_idx = torch.bitwise_not(torch.isnan(targ_p))
    targ_p = targ_p[valid_idx]
    pred_p = pred_p[valid_idx]

    return targ_p, pred_p


def build_skewed_p_loss(loss_dict):
    """
    Args:
        l_max (float): maximum possible value of the loss
        l_0 (float): maximum loss when the gap is large (i.e. p = 0)
        a (float): slope of the loss with respect to gap overestimate
                   (i.e. p underestimate)
        b (float): slope of the loss with respect to gap underestimate
                    (i.e. p overestimate)
    """

    params = loss_dict["params"]

    l_max = loss_dict["coef"]
    l_0 = params["l_0"]
    a = params["a"]
    b = params["b"]
    lower_key = loss_dict["params"]["lower_energy"]
    upper_key = loss_dict["params"]["upper_energy"]
    expec_gap = loss_dict["params"]["expected_gap"] * \
        const.AU_TO_KCAL["energy"]
    func_type = loss_dict["params"].get("func_type", "gaussian")

    factor_num = np.exp(b) * (b * (-1 + np.exp(-a))
                              + a * (-1 + np.exp(b)))
    factor_denom = a - a * np.exp(b) + b * np.exp(b) * (-1 + np.exp(a))
    factor = factor_num / factor_denom
    true_l0 = l_0 / factor

    def loss_fn(ground_truth, results, **kwargs):

        p_targ, p_pred = get_p(ground_truth=ground_truth,
                               results=results,
                               upper_key=upper_key,
                               lower_key=lower_key,
                               expec_gap=expec_gap,
                               func_type=func_type)

        loss_1 = true_l0 * torch.exp(np.log(l_max / true_l0) * p_targ)

        delta = p_pred - p_targ
        loss_2_num = np.exp(b) * (b * (-1 + torch.exp(-a * delta))
                                  + a * (-1 + torch.exp(b * delta)))
        loss_2_denom = (a - a * np.exp(b)
                        + b * np.exp(b) * (-1 + np.exp(a)))
        loss_2 = loss_2_num / loss_2_denom

        loss = (loss_1 * loss_2).mean()

        return loss

    return loss_fn


def build_zhu_loss(loss_dict):

    lower_key = loss_dict["params"]["lower_energy"]
    upper_key = loss_dict["params"]["upper_energy"]
    expec_gap = loss_dict["params"]["expected_gap"] * \
        const.AU_TO_KCAL["energy"]
    loss_key = loss_dict["params"].get("loss_type", "mse")
    func_type = loss_dict["params"].get("func_type", "gaussian")
    pos_weight = loss_dict["params"].get("pos_weight")

    if loss_key == "mse":
        loss_type = mse_operation
    elif loss_key == "cross_entropy":
        loss_type = get_cross_entropy_loss(pos_weight)

    coef = loss_dict["coef"]

    def loss_fn(ground_truth, results, **kwargs):

        targ_p, pred_p = get_p(ground_truth=ground_truth,
                               results=results,
                               upper_key=upper_key,
                               lower_key=lower_key,
                               expec_gap=expec_gap,
                               func_type=func_type)

        diff = loss_type(targ_p,
                         pred_p)
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
    func_type = loss_dict["params"].get("func_type", "gaussian")

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
            func_type=func_type,
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
                                             func_type=func_type,
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


def build_diabat_sign_loss(loss_dict):
    """
    Loss that makes the diabatic coupling switch
    signs as the molecule passes through a conical
    intersection, so that it has to be zero at some
    point. Therefore an intersection must occur.
    """

    params = loss_dict["params"]
    # the off-diagonal diabatic coupling
    off_diag_key = params["coupling"]
    # the diagonal keys whose sign you want to be the same
    # as the diabatic sign
    diag_keys = params["diagonal"]
    coef = loss_dict["coef"]

    def loss_fn(ground_truth, results, **kwargs):

        delta = (results[diag_keys[1]]
                 - results[diag_keys[0]])

        valid_idx = torch.bitwise_not(torch.isnan(delta))

        delta = delta[valid_idx]
        pred = results[off_diag_key][valid_idx]

        sign = torch.sign(delta)
        targ = torch.abs(pred) * sign

        if len(targ) == 0:
            loss = 0.0
        else:
            diff = mse_operation(targ, pred)
            loss = coef * torch.mean(diff)

        return loss

    return loss_fn


def correct_nacv_sign(pred,
                      targ,
                      num_atoms):

    targ_list = torch.split(targ, num_atoms)
    pred_list = torch.split(pred, num_atoms)
    signs = []

    for targ_batch, pred_batch in zip(targ_list,
                                      pred_list):
        pos_delta = ((targ_batch - pred_batch).abs()
                     .mean().item())
        neg_delta = ((targ_batch + pred_batch).abs()
                     .mean().item())
        sign = 1 if (pos_delta < neg_delta) else -1
        signs.append(sign)

    sign_tensor = (torch.cat([sign * torch.ones(n, 3)
                              for sign, n in zip(signs, num_atoms)])
                   .to(targ.device))

    targ = targ * sign_tensor

    return pred, targ


def build_nacv_loss(loss_dict):

    params = loss_dict["params"]
    key = params["key"]
    loss_type = params.get("loss_type", "mse")

    coef = loss_dict["coef"]

    take_abs = params.get("abs", False)
    take_max = params.get("max", False)

    def loss_fn(ground_truth,
                results,
                **kwargs):

        targ = ground_truth[key]
        pred = results[key]

        if take_abs:

            targ = abs(targ)
            pred = abs(pred)

        else:
            num_atoms = ground_truth["num_atoms"].tolist()
            pred, targ = correct_nacv_sign(pred=pred,
                                           targ=targ,
                                           num_atoms=num_atoms)

        valid_idx = torch.bitwise_not(torch.isnan(targ))
        targ = targ[valid_idx]
        pred = pred[valid_idx]

        if take_max:
            idx = abs(targ).argmax()
            targ = targ[idx]
            pred = pred[idx]

        if loss_type == "mse":
            diff = mse_operation(targ, pred)
        elif loss_type == "mae":
            diff = mae_operation(targ, pred)
        else:
            raise NotImplementedError

        loss = coef * torch.mean(diff)

        return loss

    return loss_fn


def build_trans_dip_loss(loss_dict):

    params = loss_dict["params"]
    key = params["key"]
    coef = loss_dict["coef"]

    def loss_fn(ground_truth,
                results,
                **kwargs):

        targ = ground_truth[key].reshape(-1, 3)
        pred = results[key].reshape(-1, 3)

        pos_delta = ((targ - pred) ** 2).sum(-1)
        neg_delta = ((targ + pred) ** 2).sum(-1)

        signs = (torch.ones(pos_delta.shape[0],
                            dtype=torch.long)
                 .to(pos_delta.device))
        signs[neg_delta < pos_delta] = -1
        targ = targ * signs.reshape(-1, 1)

        targ = targ.reshape(-1)
        pred = pred.reshape(-1)

        valid_idx = torch.bitwise_not(torch.isnan(targ))
        targ = targ[valid_idx]
        pred = pred[valid_idx]

        diff = mse_operation(targ, pred)
        loss = coef * torch.mean(diff)

        return loss

    return loss_fn


def name_to_func(name):
    dic = {
        "mse": build_mse_loss,
        "mae": build_mae_loss,
        "rmse": build_rmse_loss,
        "cross_entropy": build_cross_entropy_loss,
        "logits_cross_entropy": build_logits_cross_entropy_loss,
        "zhu": build_zhu_loss,
        "zhu_grad": build_zhu_grad_loss,
        "diabat_sign": build_diabat_sign_loss,
        "skewed_p": build_skewed_p_loss,
        "nacv": build_nacv_loss,
        'trans_dipole': build_trans_dip_loss
    }
    func = dic[name]
    return func


def get_all_losses(multi_loss_dict):
    loss_fns = []
    for key, loss_list in multi_loss_dict.items():
        build_fn = name_to_func(key)

        # for backwards compatability
        if key in ["mse", "mae", "rmse"]:
            loss_coef = {sub_dic["params"]["key"]:
                         sub_dic["coef"] for sub_dic in loss_list}
            cutoff = {}
            for sub_dic in loss_list:
                if 'cutoff' not in sub_dic["params"]:
                    continue
                key = sub_dic["params"]["key"]
                val = sub_dic["params"]["cutoff"]
                cutoff[key] = val

            loss_fn = build_fn(loss_coef,
                               cutoff=cutoff)
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
            this_loss = loss_fn(*args, **kwargs)
            # print(loss_fn, this_loss)
            loss += this_loss
        return loss
    return calc_loss
