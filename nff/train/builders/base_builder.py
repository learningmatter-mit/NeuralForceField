"""Code builders to help the creation of models, functions and other classes
    while checking for the validity of hyperparameters.
"""


class ParameterError(Exception):
    """Raised when a hyperparameter is of incorrect type"""
    pass


class Builder:
    """Base class to build models or building blocks from hyperparameters
    """

    @property
    def params_type(self):
        return {}

    def check_parameters(self, params):
        """Check whether the parameters correspond to the specified types

        Args:
            params (dict)
        """
        for key, val in params.items():
            try:
                if not isinstance(val, self.params_type[key]):
                    raise ParameterError(
                            '%s is not %s' % (str(key), self.params_type[key])
                 )

            except KeyError:
                pass

