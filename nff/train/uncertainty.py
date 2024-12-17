import os
import warnings
from typing import List, Tuple, Union

import numpy as np
import torch

from ..io.gmm import GaussianMixture

__all__ = [
    "Uncertainty",
    "EnsembleUncertainty",
    "EvidentialUncertainty",
    "MVEUncertainty",
    "GMMUncertainty",
    "ConformalPrediction",
]


class Uncertainty:
    def __init__(
        self,
        order: str,
        calibrate: bool,
        cp_alpha: Union[None, float] = None,
        min_uncertainty: float = None,
        *args,
        **kwargs,
    ):
        assert order in [
            "atomic",
            "system_sum",
            "system_mean",
            "system_max",
            "system_min",
            "system_mean_squared",
            "system_root_mean_squared",
        ], f"{order} not implemented"
        self.order = order
        self.calibrate = calibrate
        self.umin = min_uncertainty

        if self.calibrate:
            assert cp_alpha is not None, "cp_alpha must be specified for calibration"

            self.CP = ConformalPrediction(alpha=cp_alpha)

    def __call__(self, *args, **kwargs):
        return self.get_uncertainty(*args, **kwargs)

    def set_min_uncertainty(self, min_uncertainty: float, force: bool = False) -> None:
        """
        Set the minimum uncertainty value to be used for scaling the uncertainty.
        """
        if getattr(self, "umin") is None:
            self.umin = min_uncertainty
        elif force:
            warnings.warn(f"Uncertainty: min_uncertainty already set to {self.umin}. Overwriting.")
            self.umin = min_uncertainty
        else:
            raise Exception(f"Uncertainty: min_uncertainty already set to {self.umin}")

    def scale_to_min_uncertainty(self, uncertainty: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Scale the uncertainty to the minimum value.
        """
        if getattr(self, "umin") is not None:
            if self.order not in ["system_mean_squared"]:
                uncertainty = uncertainty - self.umin
            else:
                uncertainty = uncertainty - self.umin**2

        return uncertainty

    def fit_conformal_prediction(
        self,
        residuals_calib: Union[np.ndarray, torch.Tensor],
        heuristic_uncertainty_calib: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """
        Fit the Conformal Prediction model to the calibration data.
        """
        self.CP.fit(residuals_calib, heuristic_uncertainty_calib)

    def calibrate_uncertainty(
        self, uncertainty: Union[np.ndarray, torch.Tensor], *args, **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Calibrate the uncertainty using Conformal Prediction.
        """
        if getattr(self.CP, "qhat") is None:
            raise Exception("Uncertainty: ConformalPrediction not fitted.")

        cp_uncertainty, qhat = self.CP.predict(uncertainty)

        return cp_uncertainty

    def get_system_uncertainty(self, uncertainty: torch.Tensor, num_atoms: List[int]) -> torch.Tensor:
        """
        Get the uncertainty for the entire system.
        """
        assert "system" in self.order, f"{self.order} does not contain 'system'"

        assert len(uncertainty) == len(num_atoms), "Number of systems do not match"

        assert all([len(u) == n for u, n in zip(uncertainty, num_atoms)]), "Number of atoms in each system do not match"

        if self.order == "system_sum":
            uncertainty = uncertainty.sum(dim=-1)
        elif self.order == "system_mean":
            uncertainty = uncertainty.mean(dim=-1)
        elif self.order == "system_max":
            uncertainty = uncertainty.max(dim=-1).values
        elif self.order == "system_min":
            uncertainty = uncertainty.min(dim=-1).values
        elif self.order == "system_mean_squared":
            uncertainty = (uncertainty**2).mean(dim=-1)
        elif self.order == "system_root_mean_squared":
            uncertainty = (uncertainty**2).mean(dim=-1) ** 0.5

        return uncertainty

    def get_uncertainty(self, results: dict, *args, **kwargs):
        """
        Get the uncertainty from the results.
        """
        return NotImplementedError


class ConformalPrediction:
    """
    Copied from https://github.com/ulissigroup/amptorch
    Performs quantile regression on score functions to obtain the estimated qhat
        on calibration data and apply to test data during prediction.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(
        self,
        residuals_calib: Union[np.ndarray, torch.Tensor],
        heuristic_uncertainty_calib: Union[np.ndarray, torch.Tensor],
    ) -> None:
        # score function
        scores = abs(residuals_calib / heuristic_uncertainty_calib)
        scores = np.array(scores)

        n = len(residuals_calib)
        qhat = torch.quantile(torch.from_numpy(scores), np.ceil((n + 1) * (1 - self.alpha)) / n)
        qhat_value = np.float64(qhat.numpy()).item()
        self.qhat = qhat_value

    def predict(
        self, heuristic_uncertainty_test: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], float]:
        cp_uncertainty_test = heuristic_uncertainty_test * self.qhat
        return cp_uncertainty_test, self.qhat


class EnsembleUncertainty(Uncertainty):
    """
    Ensemble uncertainty estimation using the variance or standard deviation of the
    predictions from the model ensemble.
    """

    def __init__(
        self,
        quantity: str,
        order: str,
        std_or_var: str = "var",
        min_uncertainty: Union[float, None] = None,
        orig_unit: Union[str, None] = None,
        targ_unit: Union[str, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            order=order,
            min_uncertainty=min_uncertainty,
            calibrate=False,
            *args,
            **kwargs,
        )
        assert std_or_var in ["std", "var"], f"{std_or_var} not implemented"
        self.q = quantity
        self.orig_unit = orig_unit
        self.targ_unit = targ_unit
        self.std_or_var = std_or_var

    def convert_units(self, value: Union[float, np.ndarray], orig_unit: str, targ_unit: str):
        """
        Convert the energy/forces units of the value from orig_unit to targ_unit.
        """
        if orig_unit == targ_unit:
            return value

        conversion = {
            "eV": {"kcal/mol": 23.0605, "kJ/mol": 96.485},
            "kcal/mol": {"eV": 0.0433641, "kJ/mol": 4.184},
            "kJ/mol": {"eV": 0.0103641, "kcal/mol": 0.239006},
        }

        converted_val = value * conversion[orig_unit][targ_unit]
        return converted_val

    def get_energy_uncertainty(
        self,
        results: dict,
    ):
        """
        Get the uncertainty for the energy.
        """
        if self.orig_unit is not None and self.targ_unit is not None:
            results[self.q] = self.convert_units(results[self.q], orig_unit=self.orig_unit, targ_unit=self.targ_unit)

        if self.std_or_var == "std":
            val = results[self.q].std(-1)
        elif self.std_or_var == "var":
            val = results[self.q].var(-1)

        return val

    def get_forces_uncertainty(
        self,
        results: dict,
        num_atoms: List[int],
    ):
        """
        Get the uncertainty for the forces.
        """
        if self.orig_unit is not None and self.targ_unit is not None:
            results[self.q] = self.convert_units(results[self.q], orig_unit=self.orig_unit, targ_unit=self.targ_unit)

        splits = torch.split(results[self.q], list(num_atoms))
        stack_split = torch.stack(splits, dim=0)

        if self.std_or_var == "std":
            val = stack_split.std(-1)
        elif self.std_or_var == "var":
            val = stack_split.var(-1)

        val = torch.norm(val, dim=-1)

        if "system" in self.order:
            val = self.get_system_uncertainty(uncertainty=val, num_atoms=num_atoms)

        return val

    def get_uncertainty(self, results: dict, num_atoms: Union[List[int], None] = None, *args, **kwargs):
        if self.q == "energy":
            val = self.get_energy_uncertainty(results=results)
        elif self.q in ["energy_grad", "forces"]:
            val = self.get_forces_uncertainty(
                results=results,
                num_atoms=num_atoms,
            )
        else:
            raise TypeError(f"{self.q} not yet implemented")

        val = self.scale_to_min_uncertainty(val)

        return val


class EvidentialUncertainty(Uncertainty):
    """
    Evidential Uncertainty estimation using the Evidential Deep Learning framework.
    """

    def __init__(
        self,
        order: str = "atomic",
        shared_v: bool = False,
        source: str = "epistemic",
        calibrate: bool = False,
        cp_alpha: Union[float, None] = None,
        min_uncertainty: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            order=order,
            calibrate=calibrate,
            cp_alpha=cp_alpha,
            min_uncertainty=min_uncertainty,
            *args,
            **kwargs,
        )
        self.shared_v = shared_v
        self.source = source

    def check_params(self, results: dict, num_atoms=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Check if the parameters are present in the results, if the shapes are
        correct. If the order is "atomic" and shared_v is True, then the v
        parameter is averaged over the systems.
        """
        v = results["v"].squeeze()
        alpha = results["alpha"].squeeze()
        beta = results["beta"].squeeze()
        assert alpha.shape == beta.shape

        num_systems = len(num_atoms)
        total_atoms = torch.sum(num_atoms)
        if self.order == "atomic" and self.shared_v:
            assert v.shape[0] == num_systems and alpha.shape[0] == total_atoms
            v = torch.split(v, list(num_atoms))
            v = torch.stack(v, dim=0)
            v = v.mean(-1, keepdims=True)
            v = v.repeat_interleave(num_atoms)

        return v, alpha, beta

    def get_uncertainty(self, results: dict, num_atoms: Union[List[int], None] = None, *args, **kwargs) -> torch.Tensor:
        v, alpha, beta = self.check_params(results=results, num_atoms=num_atoms)

        if self.source == "aleatoric":
            uncertainty = beta / (alpha - 1)
        elif self.source == "epistemic":
            uncertainty = beta / (v * (alpha - 1))
        else:
            raise TypeError(f"{self.source} not implemented")

        if "system" in self.order and uncertainty.shape[0] != len(num_atoms):
            splits = torch.split(uncertainty, list(num_atoms))
            stack_split = torch.stack(splits, dim=0)

            uncertainty = self.get_system_uncertainty(uncertainty=stack_split, num_atoms=num_atoms)

        uncertainty = self.scale_to_min_uncertainty(uncertainty)

        if self.calibrate:
            uncertainty = self.calibrate_uncertainty(uncertainty)

        return uncertainty


class MVEUncertainty(Uncertainty):
    """
    Mean Variance Estimation (MVE) based uncertainty estimation.
    """

    def __init__(
        self,
        variance_key: str = "var",
        quantity: str = "forces",
        order: str = "atomic",
        min_uncertainty: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(order=order, min_uncertainty=min_uncertainty, *args, **kwargs)
        self.vkey = variance_key
        self.q = quantity

    def get_uncertainty(self, results: dict, num_atoms: Union[List[int], None] = None, *args, **kwargs) -> torch.Tensor:
        var = results[self.vkey].squeeze()
        assert results[self.q].shape[0] == var.shape[0]

        if "system" in self.order and var.shape[0] != len(num_atoms):
            splits = torch.split(var, list(num_atoms))
            stack_split = torch.stack(splits, dim=0)

            var = self.get_system_uncertainty(uncertainty=stack_split, num_atoms=num_atoms)

        var = self.scale_to_min_uncertainty(var)

        return var


class GMMUncertainty(Uncertainty):
    """
    Gaussian Mixture Model (GMM) based uncertainty estimation.
    """

    def __init__(
        self,
        train_embed_key: str = "train_embedding",
        test_embed_key: str = "embedding",
        n_clusters: int = 5,
        order: str = "atomic",
        covariance_type: str = "full",
        tol: float = 1e-3,
        max_iter: int = 100000,
        n_init: int = 1,
        init_params: str = "kmeans",
        verbose: int = 0,
        device: str = "cuda",
        calibrate: bool = False,
        cp_alpha: Union[float, None] = None,
        min_uncertainty: Union[float, None] = None,
        gmm_path: Union[str, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            order=order,
            calibrate=calibrate,
            cp_alpha=cp_alpha,
            min_uncertainty=min_uncertainty,
            *args,
            **kwargs,
        )
        self.train_key = train_embed_key
        self.test_key = test_embed_key
        self.n = n_clusters
        self.covar_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.verbose = verbose
        self.device = device

        if gmm_path is not None and os.path.exists(gmm_path):
            import pickle

            self.gmm_path = gmm_path
            with open(gmm_path, "rb") as f:
                self.gm_model = pickle.load(f)

            # Set the GMM parameters if the model is loaded
            self._set_gmm_params()

    def fit_gmm(self, Xtrain: torch.Tensor) -> None:
        """
        Fit the GMM model to the embedding of training data.
        """
        self.Xtrain = Xtrain
        self.gm_model = GaussianMixture(
            n_components=self.n,
            covariance_type=self.covar_type,
            tol=self.tol,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            verbose=self.verbose,
        )
        self.gm_model.fit(self.Xtrain.squeeze().cpu().numpy())

        # Save the fitted GMM model if gmm_path is specified
        if hasattr(self, "gmm_path") and not os.path.exists(getattr(self, "gmm_path")):
            self.gm_model.save(self.gmm_path)
            print(f"Saved fitted GMM model to {self.gmm_path}")

        # Set the GMM parameters
        self._set_gmm_params()

    def is_fitted(self) -> bool:
        """
        Check if the GMM model is fitted.
        """
        return getattr(self, "gm_model", None) is not None

    def _check_tensor(
        self,
        X: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Check if the input is a tensor and convert to torch.Tensor if not.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        X = X.squeeze().double().to(self.device)

        return X

    def _set_gmm_params(self) -> None:
        """
        Get the means, precisions_cholesky, and weights from the GMM model.
        """
        if self.gm_model is None:
            raise Exception("GMMUncertainty: GMM does not exist/is not fitted")

        self.means = self._check_tensor(self.gm_model.means_)
        self.precisions_cholesky = self._check_tensor(self.gm_model.precisions_cholesky_)
        self.weights = self._check_tensor(self.gm_model.weights_)

    def estimate_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """
        Estimate the log probability of the given embedding.
        """
        X = self._check_tensor(X)

        n_samples, n_features = X.shape
        n_clusters, _ = self.means.shape

        log_det = torch.sum(
            torch.log(self.precisions_cholesky.reshape(n_clusters, -1)[:, :: n_features + 1]),
            dim=1,
        )

        log_prob = torch.empty((n_samples, n_clusters)).to(X.device)
        for k, (mu, prec_chol) in enumerate(zip(self.means, self.precisions_cholesky)):
            y = torch.matmul(X, prec_chol) - (mu.reshape(1, -1) @ prec_chol).squeeze()
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)
        log2pi = torch.log(torch.tensor([2 * torch.pi])).to(X.device)
        return -0.5 * (n_features * log2pi + log_prob) + log_det

    def estimate_weighted_log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """
        Estimate the weighted log probability of the given embedding.
        """
        log_prob = self.estimate_log_prob(X)
        log_weights = torch.log(self.weights)
        weighted_log_prob = log_prob + log_weights

        return weighted_log_prob

    def log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """
        Log likelihood of the embedding under the GMM model.
        """
        weighted_log_prob = self.estimate_weighted_log_prob(X)

        weighted_log_prob_max = weighted_log_prob.max(dim=1).values
        # logsumexp is numerically unstable for big arguments
        # below, the calculation below makes it stable
        # log(sum_i(a_i)) = log(exp(a_max) * sum_i(exp(a_i - a_max))) = a_max + log(sum_i(exp(a_i - a_max)))
        wlp_stable = weighted_log_prob - weighted_log_prob_max.reshape(-1, 1)
        logsumexp = weighted_log_prob_max + torch.log(torch.sum(torch.exp(wlp_stable), dim=1))

        return logsumexp

    def probability(self, X: torch.Tensor) -> torch.Tensor:
        """
        Probability of the embedding under the GMM model.
        """
        logP = self.log_likelihood(X)

        return torch.exp(logP)

    def negative_log_likelihood(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative log likelihood of the embedding under the GMM model.
        """
        logP = self.log_likelihood(X)

        return -logP

    def get_uncertainty(
        self,
        results: dict,
        num_atoms: Union[List[int], None] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Get the uncertainty from the GMM model for the test embedding.
        """
        test_embedding = self._check_tensor(results[self.test_key])

        if self.is_fitted() is False:
            train_embedding = self._check_tensor(results[self.train_key])
            self.fit_gmm(train_embedding)

        uncertainty = self.negative_log_likelihood(test_embedding)

        if "system" in self.order:
            splits = torch.split(uncertainty, list(num_atoms))
            stack_split = torch.stack(splits, dim=0)

            uncertainty = self.get_system_uncertainty(uncertainty=stack_split, num_atoms=num_atoms).squeeze()

        uncertainty = self.scale_to_min_uncertainty(uncertainty)

        if self.calibrate:
            uncertainty = self.calibrate_uncertainty(uncertainty)

        return uncertainty
