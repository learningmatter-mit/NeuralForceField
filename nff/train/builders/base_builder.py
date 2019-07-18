"""Helper functions to create models, functions and other classes
    while checking for the validity of hyperparameters.
"""


class ParameterError(Exception):
    """Raised when a hyperparameter is of incorrect type"""
    pass


def check_parameters(params_type, params):
     """Check whether the parameters correspond to the specified types
 
     Args:
         params (dict)
     """
     for key, val in params.items():
         try:
             if not isinstance(val, params_type[key]):
                 raise ParameterError(
                         '%s is not %s' % (str(key), params_type[key])
              )
 
         except KeyError:
             pass
