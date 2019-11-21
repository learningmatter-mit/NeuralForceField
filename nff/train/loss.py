import torch

__all__ = ["build_mse_loss"]


def build_mse_loss(loss_coef, correspondence_keys=None):
    """
    Build the mean squared error loss function.

    Args:
        loss_coef (dict): dictionary containing the weight coefficients
            for each property being predicted.
            Example: `loss_coef = {'energy': rho, 'force': 1}`
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

    correspondence_keys = {} if (correspondence_keys is None) else correspondence_keys


    def loss_fn(ground_truth, results):
        """Calculates the MSE between ground_truth and results.

        Args:
            ground_truth (dict): e.g. `{'energy': 2, 'force': [0, 0, 0]}`
            results (dict):  e.g. `{'energy': 4, 'force': [1, 2, 2]}`

        Returns:
            loss (torch.Tensor)
        """

        assert all([k in results.keys() for k in loss_coef.keys()])
        assert all([k in [*ground_truth.keys(), *correspondence_keys.keys()] for k in loss_coef.keys()])

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
                diff = (targ - pred ) ** 2
                err_sq = coef * torch.mean(diff)
                loss += err_sq

        return loss

    return loss_fn
