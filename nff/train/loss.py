import torch

__all__ = ["build_mse_loss"]


def build_mse_loss(loss_coef):
    """
    Build the mean squared error loss function.

    Args:
        loss_coef (dict): dictionary containing the weight coefficients
            for each property being predicted.
            Example: `loss_coef = {'energy': rho, 'force': 1}`

    Returns:
        mean squared error loss function

    """

    def loss_fn(ground_truth, results):
        """Calculates the MSE between ground_truth and results.

        Args:
            ground_truth (dict): e.g. `{'energy': 2, 'force': [0, 0, 0]}`
            results (dict):  e.g. `{'energy': 4, 'force': [1, 2, 2]}`

        Returns:
            loss (torch.Tensor)
        """

        assert all([k in results.keys() for k in loss_coef.keys()])
        assert all([k in ground_truth.keys() for k in loss_coef.keys()])

        loss = 0.0
        for key, coef in loss_coef.items():

            targ = ground_truth[key]
            pred = results[key].view(targ.shape)

            # select only properties which are given
            valid_idx = 1 - torch.isnan(targ)
            targ = targ[valid_idx]
            pred = pred[valid_idx]

            if len(targ) != 0:
                diff = (targ - pred ) ** 2
                err_sq = coef * torch.mean(diff)
                loss += err_sq

        return loss

    return loss_fn
