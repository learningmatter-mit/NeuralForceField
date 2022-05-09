import warnings
import torch
import abc

'''
    Adapted from https://github.com/rtqichen/torchfunc

    [1] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. "Neural Ordinary Differential Equations." Advances in Neural Information Processing Systems. 2018. 
'''

# MetaClass Object for the different ODE Solvers
class FixedGridODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, diffeq, state, step_size=None, grid_constructor=None, **unused_kwargs):
        """Initializes ODE Integrator, setting state, time-ODE, and time-grid constructor function
        
        Args:
            diffeq: torch class that provides the differential equation to be integrated
            state: list of state vectors for one time steps (size depends on ODE and forward vs. backward pass)
            step_size (float): optional argument to construct a finer or coarser time line
            grid_constructor (function): alternative method to create the time-grid
            
        Returns:
            -
            
        Raises:
            Warning in case of unsued kwargs
            ValueError: if step_size and grid_constructor are given simultaneously
        
        """
        
        # Note: included for RK4
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('atol', None)
        
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.diffeq = diffeq
        self.state = state

        if step_size is not None and grid_constructor is None:
            self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        elif grid_constructor is None:
            self.grid_constructor = lambda t: t
        else:
            raise ValueError("step_size and grid_constructor are exclusive arguments.")

    def _grid_constructor_from_step_size(self, step_size):
        """Gives time grid constructor function for given step size
        
        Args:
            step_size (float): time step
           
        Returns:
            _grid_constructor: function constructing grid
            
        Raises:
            -
        """

        def _grid_constructor(t):
            """constructs time grid for given interval and step size
            
            Args:
                t (torch.Tensor): array of time steps
                
            Returns:
                t_infer (torch.Tensor): array of inferred time steps 
                
            Raises:
                -
            
            """
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters).to(t) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @property
    @abc.abstractmethod
    def order(self):
        pass

    @abc.abstractmethod
    def step_func(self, diffeq, dt, state):
        """Step function for a single time step, to be filled by the actual realization of the Integrator
        
        Args:
            diffeq: torch class that provides the differential equation to be integrated
            t: array of time points
            dt (float): time step 
            state: tuple of state vectors for one time steps (size depends on ODE and forward vs. backward pass)
            
        Returns:
            will return the step in state variables
        """ 
        pass

    def integrate(self, t):
        """Integrator performing all time steps
        
        Args:
            t: array of time points
            
        Returns:
            time series (tuple) of state vectors (torch vectors)
            
        Raises:
            AssertionError: if timeline is not strictly ascending
            AssertionError: if ends of the time grid do not match the time array end points
        
        """
        _assert_increasing(t)
        t = t.type_as(self.state[0])
        time_grid = self.grid_constructor(t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(self.state[0])

        solution = [self.state]

        state = self.state
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            step_state = self.step_func(self.diffeq, dt, state)
            new_state = tuple(state_ + step_ for state_, step_ in zip(state, step_state))
            state = new_state

            solution.append(new_state)

        return tuple(map(torch.stack, tuple(zip(*solution))))  
    
    
def _check_inputs(diffeq, state, t):
    """Makes sure that state vectors are a tuple, diffeq returns a tuple, if time decreases reverses direction of diffeq
    
    Args:
        diffeq (nn.module): function that yields acceleration and velocoties
        state: tuple of state vectors for one time step
        t: time series
        
    Returns:
        tensor_input (bool): whether inputs were tensors instead of tuples
        diffeq: updated
        state: updated
        t: updated
        
    Raises:
        AssertionError: if elements of state are not type torch.Tensor
        TypeError: if state is not a floating point tensor
        TypeError: if t is not a floating point tensor
    
    """
    # check for tensor vs tuple type input
    tensor_input = False
    if torch.is_tensor(state):
        tensor_input = True
        state = (state,)
        _base_nontuple_func_ = diffeq
        diffeq = lambda state: (_base_nontuple_func_(state[0]),)
    assert isinstance(state, tuple), 'The state vecotrs must be either a torch.Tensor or a tuple'
    for s_ in state:
        assert torch.is_tensor(s_), 'each element must be a torch.Tensor but received {}'.format(type(s_))

    # time reversal of the ODE
    if _decreasing(t):
        t = -t
        _base_reverse_func = diffeq
        diffeq = lambda state: tuple(-f_ for f_ in _base_reverse_func(state))

    # check for floating point 
    for s_ in state:
        if not torch.is_floating_point(s_):
            raise TypeError('state vector `state` must be a floating point Tensor but is a {}'.format(s_.type()))
    if not torch.is_floating_point(t):
        raise TypeError('time series `t` must be a floating point Tensor but is a {}'.format(t.type()))

    return tensor_input, diffeq, state, t  


def _decreasing(t):
    """Checks if all time points are decreasing
    
    Args:
        t: time series
        
    Returns:
        bool: True if t is strictly decreasing, otherwise False
        
    Raises:
        -
    
    """
    return (t[1:] < t[:-1]).all()   
    
    
    
def _handle_unused_kwargs(solver, unused_kwargs):
    """Returns warning if there are unused arguments
    
    Args:
        unused_kwargs
        
    Returns:
        -
        
    Raises:
        Warning
    """
    
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))    

        
def _assert_increasing(t):
    """Raises AssertionError if time series is not strictly increasing
    
    Args:
        t: time series
        
    Returns:
        -
        
    Raises:
        AssertionError
    """
    assert (t[1:] > t[:-1]).all(), 't must be strictly increasing or decrasing'
    
    
def _flatten(sequence):
    """Returns flattened torch tensor of parameters of diffeq object
    
    Args:
        sequence of parameters of a diffeq object
        
    Returns:
        torch tensor of given parameters
        
    Raises:
        -
    """
    flat = [element.contiguous().view(-1) for element in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    """Flattens the gradient with regards to parameters, 
    if that doesn't exist it returns zeros instead
    
    Args:
        sequence: autograd gradients 
        like_sequence: flattened parameter sequence
        
    Returns:
        flattened parameter gradients as torch.tensor
    
    Raises: 
        -
    """
    flat = [
        elem_seq.contiguous().view(-1) if elem_seq is not None else torch.zeros_like(elem_like_seq).view(-1)
        for elem_seq, elem_like_seq in zip(sequence, like_sequence)
    ]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])
