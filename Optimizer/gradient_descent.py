import numpy as np
from wzk import mp_wrapper

import Optimizer.objectives as obj
import Optimizer.path as path


# Gradient Descent
def gradient_descent_wrapper_mp(x, fun, grad, par):
    def gd_wrapper(__x):
        return gradient_descent(x=__x, fun=fun, grad=grad, par=par)

    return mp_wrapper(x, n_processes=par.n_processes, fun=gd_wrapper)


def __get_adaptive_step_size(x, x_old, dcost_dx, dcost_dx_old,
                             step_size):
    # Calculate the adaptive step shape
    if x_old is None:
        ss = step_size
    else:
        diff_x = np.abs(x - x_old)
        diff_grad = np.abs(dcost_dx - dcost_dx_old)
        diff_grad_squared_sum = (diff_grad ** 2).sum(axis=(-2, -1))
        const_ss_idx = diff_grad_squared_sum == 0
        diff_grad_squared_sum[const_ss_idx] = 1

        ss = (diff_x * diff_grad).sum(axis=(-2, -1)) / diff_grad_squared_sum
        ss[const_ss_idx] = step_size
        if np.size(ss) > 1:
            ss = ss.reshape((ss.size,) + (1,) * (dcost_dx.ndim - 1))

    return ss


def gradient_descent(*, x, fun, grad,
                     par):
    # Gradient Descent Loop
    dcost_dx_old = None
    x_old = None

    # If the parameters aren't given for all steps, expand them
    if np.size(par.grad_tol) == 1:
        par.grad_tol = np.full(par.n_steps, fill_value=float(par.grad_tol))

    if np.size(par.hesse_weighting) == 1:
        par.hesse_weighting = np.full(par.n_steps, fill_value=float(par.hesse_weighting))

    grad_max_evolution = []
    if par.return_x_list:
        x_list = np.zeros(x.shape + (par.n_steps,))
    else:
        x_list = None

    for i in range(par.n_steps):

        dcost_dx = grad(x=x, i=i)

        # Correct via an approximated hesse function
        if par.hesse_inv is not None and par.hesse_weighting[i] > 0:
            # one possibility is to use relative motions, or to unravel the interval [0,360]
            d_cost__d_x_hesse = (par.hesse_inv[np.newaxis, ...] @ dcost_dx.reshape(-1, par.hesse_inv.shape[-1], 1)
                                 ).reshape(dcost_dx.shape)
            dcost_dx = dcost_dx * (1 - par.hesse_weighting[i]) + d_cost__d_x_hesse * par.hesse_weighting[i]

        if par.callback is not None:
            dcost_dx = par.callback(x=x.copy(), jac=dcost_dx.copy())  # , count=o) -> callback function handles count

        # Calculate the adaptive step shape
        if par.adjust_step_size:
            ss = __get_adaptive_step_size(x=x, x_old=x_old,
                                          dcost_dx=dcost_dx, dcost_dx_old=dcost_dx_old,
                                          step_size=par.step_size)
            x_old = x.copy()
            dcost_dx_old = dcost_dx.copy()
        else:
            ss = par.step_size

        dcost_dx *= ss

        # Cut gradient that no step is larger than a tolerance value
        max_grad = np.abs(dcost_dx).max(axis=(1, 2))
        grad_too_large = max_grad > par.grad_tol[i]
        dcost_dx[grad_too_large] /= max_grad[grad_too_large, np.newaxis, np.newaxis] / par.grad_tol[i]

        # Apply gradient descent
        x -= dcost_dx

        if par.return_x_list:
            x_list[..., i] = x

        # Clip the values to fit with the range of values
        if par.prune_limits is not None:
            x = par.prune_limits(x)

    objective = fun(x=x)

    if par.return_x_list:
        return x, objective, x_list
    else:
        return x, objective


# Trajectory
def solve_chomp(*, q_inner, q_start, q_end,
                par, gd):
    """
    Perform gradient descent steps for multiple paths on one world
    """

    weighting = par.weighting.copy()

    def x_inner2x_mp(x):
        return path.x_inner2x(x_inner=x, x_start=q_start, x_end=q_end)

    def fun(x):
        return obj.chomp_cost(q=x_inner2x_mp(x=x), par=par)

    def grad(x, i):
        par.weighting = weighting.at_idx(i=i)
        return obj.chomp_grad(q=x_inner2x_mp(x=x), par=par, jac_only=True)

    gd.prune_limits = par.robot.prune_joints2limits
    x_ms, objective = gradient_descent_wrapper_mp(q_inner, fun=fun, grad=grad, par=gd)

    par.weighting = weighting.copy()

    return x_ms, objective


# Multi start wrapper
def solve_chomp_ms(*, q0, q_start, q_end,
                   par, gd,
                   return_all=False, verbose=0):

    if return_all:
        x_initial = path.x_inner2x(x_inner=q0.copy(), x_start=q_start, x_end=q_end)
    else:
        x_initial = None

    weighting = par.weighting.copy()
    q0, objective = solve_chomp(q_inner=q0, q_start=q_start, q_end=q_end, par=par, gd=gd)
    par.weighting = weighting

    objective[np.isnan(objective)] = np.inf

    if return_all:
        x_ms = path.x_inner2x(x_inner=q0, x_start=q_start, x_end=q_end)
        return x_initial, x_ms, objective

    else:
        return q0, objective
