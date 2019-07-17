import torch


__all__ = ["build_mse_loss"]


def build_mse_loss(rho=1):
    """
    Build the mean squared error loss function.

    Args:
        rho (float): parameter multiplying the second argument
            factor

    Returns:
        mean squared error loss function

    """
    loss_coef = [1, rho]

    def loss_fn(ground_truth, predictions):
        loss = 0.0
        for targ, pred, coef  in zip(ground_truth, predictions, loss_coef):
            diff = (targ - pred) ** 2
            err_sq = coef * torch.mean(diff)
            loss += err_sq
        return loss

    return loss_fn
