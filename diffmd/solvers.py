from diffmd.solver_base import FixedGridODESolver
from diffmd.solver_base import _check_inputs, _flatten, _flatten_convert_none_to_zeros

import torch
from torch import nn

class VelVerlet_NVE(FixedGridODESolver):
    """Velocity Verlet updater for NVE ODE forward and backward
    """

    def step_func(self, diffeq, dt, state):
        """Propagates state vectors and ajoints one step in time
        
        Args:
            diffeq: simple ODE for forward step and augmented ODE for backward step
            dt (float): time step
            state (tuple): state vectors as well as adjoints and dL_dparams
            
        Returns:
            increment to propagate state by 1 dt
            
        Raises:
            -
        """
        NUM_VAR = 2 # vels and coords for NVE

        if len(state) == NUM_VAR: # integrator in the forward call 
            dvdt_0, dqdt_0 = diffeq(state)

            v_step_half = 1/2 * dvdt_0 * dt 
 
            q_step_full = (state[0] + v_step_half) * dt 

            # gradient full at t + dt 
            dvdt_full, dqdt_half = diffeq((state[0] + v_step_half, state[1] + q_step_full))
 
            v_step_full = v_step_half + 1/2 * dvdt_full * dt

            return tuple((v_step_full, q_step_full))
        
        elif len(state) == NUM_VAR * 2 + 1: # integrator in the backward call 
            # diffeq is the automatically generated ODE for adjoints (returns more than the original forward ODE)
            dvdt_0, dqdt_0, v_adj_0, q_adj_0, dLdpar_0  = diffeq(state) 

            # more importantly are there better way to integrate the adjoint state other than midpoint integration 

            v_step_half = 1/2 * dvdt_0 * dt 

            q_step_full = (state[0] + v_step_half) * dt 

            # half step adjoint update 
            vadjoint_half = v_adj_0 * 0.5 * dt 
            qadjoint_half = q_adj_0 * 0.5 * dt 
            dLdpar_half   = dLdpar_0 * 0.5 * dt 

            dvdt_mid, dqdt_mid, v_adj_mid, q_adj_mid, dLdpar_mid = diffeq(
                (state[0] + v_step_half, state[1] + q_step_full, 
                 state[2] + vadjoint_half, state[3] + qadjoint_half, 
                 state[4] + dLdpar_half))

            v_step_full = v_step_half + 1/2 * dvdt_mid * dt 

            # half step adjoint update 
            vadjoint_step = v_adj_mid * dt 
            qadjoint_step = q_adj_mid * dt  
            dLdpar_step   = dLdpar_mid * dt         

            return (v_step_full, q_step_full, vadjoint_step, qadjoint_step, dLdpar_step)
        else:
            raise ValueError("received {} argumets integration, but should be {} for the forward call or {} for the backward call".format(len(state), NUM_VAR, 2 * NUM_VAR + 1))


class VelVerlet_NHC(FixedGridODESolver):
    """Velocity Verlet updater for Nosé-Hoover-Chains ODE forward and backward
    """

    def step_func(self, diffeq, dt, state):
        """Propagates state vectors and ajoints one step in time
        
        Args:
            diffeq: simple ODE for forward step and augmented ODE for backward step
            dt (float): time step
            state (tuple): state vectors as well as adjoints and dL_dparams
            
        Returns:
            increment to propagate state by 1 dt
            
        Raises:
            -
        """            
        NUM_VAR = 3 # vels, coords, p_eta's for Nose-Hoover Chains

        if len(state) == NUM_VAR: # integrator in the forward call 
            dvdt_0, dqdt_0, dpeta_dt_0 = diffeq(state)

            # update half step 
            v_step_half = 0.5 *  dvdt_0 * dt 
            peta_step_half = 0.5 * dpeta_dt_0 * dt

            #get full step in positions 
            q_step_full = (state[0] + v_step_half) * dt 

            # gradient full at t + dt 
            dvdt_full, dqdt_half, dpeta_dt_half = diffeq((state[0] + v_step_half, state[1] + q_step_full, state[2] + peta_step_half))

            # full step update 
            v_step_full = v_step_half + 0.5 * dvdt_full * dt
            peta_step_full = peta_step_half + 0.5 * dpeta_dt_half * dt 

            return tuple((v_step_full, q_step_full, peta_step_full))

        elif len(state) == NUM_VAR * 2 + 1: # integrator in the backward call 
            dvdt_0, dqdt_0, dpeta_dt_0, v_adj_0, q_adj_0, peta_adj_0, dLdpar_0 = diffeq(state)

            v_step_half = 0.5 * dvdt_0 * dt 

            peta_step_half = 0.5 * dpeta_dt_0 * dt 

            q_step_full = (ystate[0] + v_step_half) * dt 

            # half step adjoint update 
            v_adj_half    = v_adj_0 * 0.5 * dt 
            q_adj_half    = q_adj_0 * 0.5 * dt 
            peta_adj_half = peta_adj_0 * 0.5 * dt
            dLdpar_half   = dLdpar_0 * 0.5 * dt 

            dvdt_mid, dqdt_mid, dpeta_dt_mid, v_adj_mid, q_adj_mid, peta_adj_mid, dLdpar_mid = diffeq(
                (state[0] + v_step_half, state[1] + q_step_full, state[2] + pveta_step_half, 
                 state[3] + v_adj_half, state[4] + q_adj_half, state[5] + peta_adj_half, 
                 state[6] + dLdpar_half
                       ))

            v_step_full  = v_step_half + 0.5 * dvdt_mid * dt 
            peta_step_full = peta_step_half + 0.5 * dpeta_dt_mid * dt 

            # half step adjoint update 
            v_adj_step    = v_adj_mid * dt # update adjoint state 
            q_adj_step    = q_adj_mid * dt 
            peta_adj_step = peta_adj_mid * dt
            dLdpar_step   = dLdpar_mid * dt         

            return (v_step_full, q_step_full, peta_step_full, 
                    v_adj_step, q_adj_step, peta_adj_step,
                    dLdpar_step)

        else:
            raise ValueError("received {} argumets integration, but should be {} for the forward call or {} for the backward call".format(
                    len(y), NUM_VAR, 2 * NUM_VAR + 1))    
            

    
def odeint(diffeq, state, t, method=None, options=None):
    """Calls the correct integrator for the specific ODE, performs sanity checks
    
    Args:
        diffeq (nn.module): function that yields acceleration and velocoties
        state: tuple of state vectors for one time step
        t: time series
        method (string): specifies the solver needed for ODE
        
    Returns:
        solution: time series generated from starting condition and given ODE+solver
        
    Raises:
        ValueError: If method was not specified
        from _check_inputs
        AssertionError: if elements of state are not type torch.Tensor
        TypeError: if state is not a floating point tensor
        TypeError: if t is not a floating point tensor
    """

    SOLVERS = {
    'NVE': VelVerlet_NVE, 
    'NHC': VelVerlet_NHC
    }

    tensor_input, diffeq, state, t = _check_inputs(diffeq, state, t)

    if options is None:
        options = {}
    elif method is None:
        raise ValueError('Cannot supply `options` without specifying `method`')

    if method is None:
        raise ValueError('Method needs to be specified!')

    solver = SOLVERS[method](diffeq, state, **options)
    solution = solver.integrate(t)
    if tensor_input:
        solution = solution[0]
    return solution  


def odeint_adjoint(diffeq, state, t, method=None, options=None):
    """Wrapper function calling on the complex forward/backward pass of the adjoint method, insures that everything is handled as tuples
    
    Args:
        diffeq (nn.module): function that yields acceleration and velocoties
        state: tuple of state vectors for one time step
        t: time series
        method (string): specifies the solver needed for ODE
    
    Returns:
        expanded state trajectory 
        
    Raises:
        TypeError: if diffeq is not an instance of nn.Module
        TypeError: if state vectors are not bundled as tuple
        ValueError: if method is not supplied
    """
        
    if not isinstance(diffeq, nn.Module):
        raise TypeError('diffeq is required to be an instance of nn.Module.')

    if torch.is_tensor(state):
        raise TypeError('The state vectors have to be given as tuples of torch.Tensor`s!')
        
    if method is None:
        raise ValueError('Method needs to be specified!')

    flat_params = _flatten(diffeq.parameters())
    expanded_states = OdeintAdjointMethod.apply(*state, diffeq, t, flat_params, method, options)

    return expanded_states


class OdeintAdjointMethod(torch.autograd.Function):
    """Expanded torch.autograd class, can perform forward and backward pass in time, 
    shares parameters and information via the autograd context (ctx)
    """

    @staticmethod
    def forward(ctx, *args):
        """Forward pass in time
        
        Args:
            state: expanded tuple of state vectors
            diffeq: diffeq (nn.module): function that yields acceleration and velocoties
            t (torch.Tensor): time line
            flate_params: torch.Tensor of diffeq parameters
            method (string): specifying the fitting integrator for diffeq
            options (dict): options for ODE solver
            
        Returns:
            ans: integrated ODE
            
        Raises:
            -
        """
        state, diffeq, t, flat_params, method, options = \
            args[:-5], args[-5], args[-4], args[-3], args[-2], args[-1]

        ctx.diffeq, ctx.method, ctx.options = diffeq, method, options

        with torch.no_grad():
            traj = odeint(diffeq, state, t, method=method, options=options)
        ctx.save_for_backward(t, flat_params, *traj)
        return traj

    @staticmethod
    def backward(ctx, *grad_output):

        t, flat_params, *traj = ctx.saved_tensors
        traj = tuple(traj)
        diffeq, method, options = ctx.diffeq, ctx.method, ctx.options
        n_statevecs = len(traj)
        f_params = tuple(diffeq.parameters()) #JD: why do we need this if we have flat_params?

        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(state_aug):
            """
            Args:
                t (torch.Tensor): array of time points
                state_aug (tuple): augmented tuple of state vectors, contains info from forward and backward passes
                
            Returns:
                *time_derivatives: expanded tuple of time derivatives yielded by diffeq
                *grad_state: expanded tuple of gradients of time derivative wrt to state vectors
                grad_params: gradients of time derivative wrt to diffeq parameters
            
            Raises:
                -
            """
            state, adj_state = state_aug[:n_statevecs], state_aug[n_statevecs:2 * n_statevecs]  

            with torch.set_grad_enabled(True):
                state = tuple(statevec_.detach().requires_grad_(True) for statevec_ in state)
                #JD: right now, no diffeq depends on t, 
                #if this changes then we need gradients with respect to t as well, 
                #compare with original code
                time_derivatives = diffeq(state)
                grad_wrt_state_and_params = torch.autograd.grad(time_derivatives, state + f_params,
                    tuple(-adj_statevec_ for adj_statevec_ in adj_state), allow_unused=True, retain_graph=True)
                    
            grad_state = grad_wrt_state_and_params[:n_statevecs]
            grad_params = grad_wrt_state_and_params[n_statevecs:]

            # autograd.grad returns None if no gradient, set to zero.
            grad_state = tuple(torch.zeros_like(state_) if grad_statevec_ is None else grad_statevec_ for grad_statevec_, statevec_ in zip(grad_state, state))
            if len(f_params) == 0:
                grad_params = torch.tensor(0.).to(grad_state[0])
            else:
                grad_params = _flatten_convert_none_to_zeros(grad_params, f_params)

            return (*time_derivatives, *grad_state, grad_params)
        
        
        n_time_steps = traj[0].shape[0]
        with torch.no_grad():
            adj_state = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []
            for i in range(n_time_steps - 1, 0, -1):

                state_i = tuple(statevectraj_[i] for statevectraj_ in traj)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                timederiv_i = diffeq(state_i)

                # JD: I think this is: a(t) dot df/d\theta
                dLd_cur_t = sum(
                    torch.dot(timederiv_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    for timederiv_i_, grad_output_i_ in zip(timederiv_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if adj_params.numel() == 0:
                    adj_params = torch.tensor(0.).to(adj_state[0])
                aug_state = (*state_i, *adj_state, adj_params)
                aug_traj = odeint(augmented_dynamics, aug_state,
                    torch.tensor([t[i], t[i - 1]]), method=method, options=options)

                # Unpack aug_traj
                adj_state = aug_traj[n_statevecs:2 * n_statevecs]
                adj_params = aug_traj[2 * n_statevecs]

                adj_state = tuple(adj_statevec_[1] if len(adj_statevec_) > 0 else adj_statevec_ for adj_statevec_ in adj_state)
                if len(adj_params) > 0: adj_params = adj_params[1]

                adj_state = tuple(adj_statevec_ + grad_output_[i - 1] for adj_statevec_, grad_output_ in zip(adj_state, grad_output))

                del aug_state, aug_traj
                
            time_vjps.append(adj_time)     
            time_vjps = torch.cat(time_vjps[::-1])
            return (*adj_state, None, time_vjps, adj_params, None, None, None, None, None)
