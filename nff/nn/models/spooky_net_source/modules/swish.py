import torch
import torch.nn as nn


class Swish(nn.Module):
    """
    Swish activation function with learnable feature-wise parameters:
    f(x) = alpha*x * sigmoid(beta*x)
    sigmoid(x) = 1/(1 + exp(-x))
    For beta -> 0  : f(x) -> 0.5*alpha*x
    For beta -> inf: f(x) -> max(0, alpha*x)

    Arguments:
        num_features (int):
            Dimensions of feature space.
        initial_alpha (float):
            Initial "scale" alpha of the "linear component".
        initial_beta (float):
            Initial "temperature" of the "sigmoid component". The default value
            of 1.702 has the effect of initializing swish to an approximation
            of the Gaussian Error Linear Unit (GELU) activation function from
            Hendrycks, Dan, and Gimpel, Kevin. "Gaussian error linear units
            (GELUs)."
    """

    def __init__(
        self, num_features: int, initial_alpha: float = 1.0, initial_beta: float = 1.702
    ) -> None:
        """ Initializes the Swish class. """
        super(Swish, self).__init__()
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.register_parameter("alpha", nn.Parameter(torch.Tensor(num_features)))
        self.register_parameter("beta", nn.Parameter(torch.Tensor(num_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialize parameters alpha and beta. """
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate activation function given the input features x.
        num_features: Dimensions of feature space.

        Arguments:
            x (FloatTensor [:, num_features]):
                Input features.

        Returns:
            y (FloatTensor [:, num_features]):
                Activated features.
        """
        return self.alpha * x * torch.sigmoid(self.beta * x)
