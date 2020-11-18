import numpy as np
from wzk import angle2minuspi_pluspi


def x_flat2x(*, x_flat, n_wp=None, n_dof):
    """
    x_flat -> x
    Force the input in matrix form with n_dim column vectors for x1, x2, ... (x, y, z, ...) .
    If the input is already in this (n, n_dim) form, nothing is changed.
    """

    n_samples = x_flat.shape[0]
    if n_wp is None:
        n_wp = x_flat.size // n_dof // n_samples
    return x_flat.reshape((n_samples, n_wp, n_dof))


def x2x_flat(*, x):
    n_samples = x.shape[0]
    return x.reshape((n_samples, -1))


# REPRESENTATION - Inner
def x2x_inner(x):
    return x[..., 1:-1, :]


def x_inner2x(*, x_inner, x_start=None, x_end=None):
    n_samples = x_inner.shape[0]

    def __repeat(x):
        return x.repeat(n_samples // x.shape[0], axis=0)

    if x_start is not None:
        x_start = __repeat(x_start)
        x_inner = np.concatenate((x_start, x_inner), axis=-2)

    if x_end is not None:
        x_end = __repeat(x_end)
        x_inner = np.concatenate((x_inner, x_end), axis=-2)

    return x_inner


def q2x_q(xq, n_dim):
    x = xq[..., :n_dim]
    q = xq[..., n_dim:]
    return x, q


def x_q2q(x, q):
    """
    Root coordinates followed by the joint angles
    """

    xq = np.concatenate((x, q), axis=-1)
    return xq


# PATH PROPERTIES AND VALUES
def get_n_waypoints(*, x, n_dof, only_inner=False):
    n_samples = x.shape[0]
    n_waypoints = (x.size // n_dof) // n_samples
    if only_inner:
        return n_waypoints - 2
    else:
        return n_waypoints


def get_start_end(x, waypoints_dim=True):
    if waypoints_dim:
        return x[..., :1, :], x[..., -1:, :]
    else:
        return x[..., 0, :], x[..., -1, :]


def inf_joint_wrapper(x, inf_joints=None):
    if inf_joints is not None:
        x[..., inf_joints] = angle2minuspi_pluspi(x[..., inf_joints])
    return x


def get_start_end_normalization(*, q_start, q_end, n_wp,
                                joint_weighting=None, infinity_joints,
                                eps=0.01):  # Weighting for path, minimal distance in matrix, rad
    """
    Get minimal length cost between x_start and x_end with n_wp waypoints (linear connection)
    Divide the connection in (n_wp-1) equal steps, square each step and consider their sum
    """

    # norm = np.linalg.norm(x_end - x_start, axis=-1) / (n_wp-1)
    # norm = norm ** 2 * (n_wp - 1)
    # norm = np.linalg.norm(x_end - x_start, axis=-1).ravel()

    if joint_weighting is None:
        joint_weighting = 1
    else:
        joint_weighting = joint_weighting

    x_diff = (inf_joint_wrapper(x=q_end[..., 0, :] - q_start[..., 0, :], inf_joints=infinity_joints))
    x_diff = (x_diff + eps) ** 2
    # norm = 0.5 * np.sqrt((joint_weighting * x_diff).sum(axis=-1))  # Weighted sum over the joints
    norm = 0.5 * (joint_weighting * x_diff).sum(axis=-1)  # Weighted sum over the joints
    norm /= (n_wp - 1)
    # norm[:] = 1
    return norm


def get_x_steps(x,
                steps=None):
    """

    :param x:
    :param steps:  Optional (reuse)
    :return:
    """
    if steps is None:
        return np.diff(x, axis=1)
    else:
        return steps


def get_step_lengths(*, x=None,
                     steps=None,  # Optional (reuse)
                     step_lengths=None):
    if step_lengths is None:
        steps = get_x_steps(x, steps=steps)
        return np.linalg.norm(steps, axis=-1)
    else:
        return step_lengths


def trajectory_length(x=None,
                      step_lengths=None,  # Optional (reuse)
                      squared=False):  # Options
    """
    Calculate the length of the path by summing up all individual steps  (see path.step_lengths).
    Assume linear connecting between way points.
    If boolean 'squared' is True: take the squared distance of each step -> enforces equally spaced steps.
    """

    step_lengths = get_step_lengths(x=x, step_lengths=step_lengths)
    if squared:
        step_lengths **= 2
    return step_lengths.sum(axis=-1)


def linear_distance(*, x_start=None, x_end=None,
                    x=None):
    if x_start is None:
        x_start = x[..., 0, :]
    if x_end is None:
        x_end = x[..., -1, :]

    return np.linalg.norm(x_end - x_start, axis=-1)


def get_x_substeps(*, x, n_substeps,
                   steps=None,
                   infinity_joints=None,
                   include_end_point=True):
    """
    Calculate the substeps between the neighboring way points for a given number of substeps.
    'include_end' by default the last way point is included in the result.
    """

    n_samples, n_wp, n_dof = x.shape

    # The only fill in substeps if the the number is greater 1,
    if n_substeps <= 1 or n_wp <= 1:
        return x

    steps = get_x_steps(x=x, steps=steps)
    steps = inf_joint_wrapper(x=steps, inf_joints=infinity_joints)

    # Create an array witch contains all substeps in one go
    x_substep_i = steps / n_substeps
    delta = np.arange(n_substeps) * x_substep_i[..., np.newaxis]
    x_ss = x[..., :-1, :, np.newaxis] + delta

    x_ss = x_ss.transpose((0, 1, 3, 2)).reshape((n_samples, (n_wp - 1) * n_substeps, n_dof))

    if include_end_point:
        x_ss = x_inner2x(x_inner=x_ss, x_start=None, x_end=x[..., -1:, :])

    x_ss[..., infinity_joints] = angle2minuspi_pluspi(x_ss[..., infinity_joints])
    return x_ss


def linear_connection(q, n_waypoints, infinity_joints, weighting=None):

    _, n_points, n_dof = q.shape
    n_connections = n_points - 1
    x_rp_steps = get_x_steps(x=q)
    x_rp_steps = inf_joint_wrapper(x=x_rp_steps, inf_joints=infinity_joints)
    x_rp_steps = x_rp_steps[0]
    if weighting is not None:
        x_rp_steps *= weighting

    # Distribute the waypoints equally along the linear sequences of the initial path
    x_rp_steps_norm = np.linalg.norm(x_rp_steps, axis=-1)
    if np.sum(x_rp_steps_norm) == 0:
        x_rp_steps_relative_length = np.full(n_connections, fill_value=1 / n_connections)
    else:
        x_rp_steps_relative_length = x_rp_steps_norm / np.sum(x_rp_steps_norm)

    # Adjust the number of waypoints for each step to make the initial guess as equally spaced as possible
    n_wp_sub_exact = x_rp_steps_relative_length * (n_waypoints - 1)
    n_wp_sub = np.round(n_wp_sub_exact).astype(int)
    n_wp_sub_acc = n_wp_sub_exact - n_wp_sub

    n_waypoints_diff = (n_waypoints - 1) - np.sum(n_wp_sub)

    # If the waypoints do not match, change the substeps where the rounding was worst
    if n_waypoints_diff != 0:
        n_wp_sub_acc = 0.5 + np.sign(n_waypoints_diff) * n_wp_sub_acc
        steps_corrections = np.argsort(n_wp_sub_acc)[-np.abs(n_waypoints_diff):]
        n_wp_sub[steps_corrections] += np.sign(n_waypoints_diff)

    n_wp_sub_cs = n_wp_sub.cumsum()
    n_wp_sub_cs = np.hstack((0, n_wp_sub_cs)) + 1

    # Add the linear interpolation between the random waypoints step by step for each dimension
    x_path = np.zeros((1, n_waypoints, n_dof))
    x_path[:, 0, :] = q[:, 0, :]
    for i_rp in range(n_connections):
        x_path[:, n_wp_sub_cs[i_rp]:n_wp_sub_cs[i_rp + 1], :] = \
            get_x_substeps(x=q[:, i_rp:i_rp + 2, :], n_substeps=n_wp_sub[i_rp],
                           include_end_point=True, infinity_joints=infinity_joints)[:, 1:, ...]

    x_path = inf_joint_wrapper(x=x_path, inf_joints=infinity_joints)
    return x_path


# ADDITIONAL FUNCTIONS
def order_path(x, x_start=None, x_end=None, weights=None):
    """
    Order the points given by 'x' [2d: (n, n_dof)] according to a weighted euclidean distance
    so that always the nearest point comes next.
    Start with the first point in the array and end with the last if 'x_start' or 'x_end' aren't given.
    """

    n_dof = None
    # Handle different input configurations
    if x_start is None:
        x_start = x[..., 0, :]
        x = np.delete(x, 0, axis=-2)
    else:
        n_dof = np.size(x_start)

    if x is None:
        n_waypoints = 0
    else:
        n_waypoints, n_dof = x.shape[-2:]

    if x_end is None:
        xi_path = np.zeros((1, n_waypoints + 1, n_dof))
    else:
        xi_path = np.zeros((1, n_waypoints + 2, n_dof))
        xi_path[0, -1, :] = x_end.ravel()

    xi_path[0, 0, :] = x_start.ravel()

    # Order the points, so that always the nearest is visited next, according to the euclidean distance
    for i in range(n_waypoints):
        x_diff = np.linalg.norm(x - x_start, axis=-1)
        i_min = np.argmin(x_diff)
        xi_path[..., 1 + i, :] = x[..., i_min, :]
        x_start = x[..., i_min, :]
        x = np.delete(x, i_min, axis=-2)

    return xi_path


#
# DERIVATIVES
def d_step_length__dx(*, x=None,  # Either
                      steps=None, step_lengths=None  # Or, - Give to speed up computation
                      ):

    steps = get_x_steps(x=x, steps=steps)
    step_lengths = get_step_lengths(steps=steps, step_lengths=step_lengths)

    jac = steps.copy()
    motion_idx = step_lengths != 0  # All steps where there is movement between t, t+1
    jac[motion_idx, :] = jac[motion_idx, :] / step_lengths[motion_idx][..., np.newaxis]

    return jac


def d_substeps__dx(n_substeps, order=0):
    """
    Get the dependence of substeps (') on the outer way points (x).
    The substeps are placed linear between the waypoints.
    To prevent counting points double one step includes only one of the two endpoints
    This gives a symmetric number of steps but ignores either the start or the end in
    the following calculations.

         x--'--'--'--x---'---'---'---x---'---'---'---x--'--'--'--x--'--'--'--x
    0:  {>         }{>            } {>            } {>         }{>         }
    1:     {         <} {            <} {            <}{         <}{         <}

    Ordering of the waypoints into a matrix:
    0:
    s00 s01 s02 s03 -> step 0 (with start)
    s10 s11 s12 s13
    s20 s21 s22 s23
    s30 s31 s32 s33
    s40 s41 s42 s43 -> step 4 (without end)

    1: (shifting by one, and reordering)
    s01 s02 s03 s10 -> step 0 (without start)
    s11 s12 s13 s20
    s21 s22 s23 s30
    s31 s32 s33 s40
    s41 s42 s43 s50 -> step 4 (with end)


    n          # way points
    n-1        # steps
    n_s        # intermediate points per step
    (n-1)*n_s  # substeps (total)

    n_s = 5
    jac0 = ([[1. , 0.8, 0.6, 0.4, 0.2],  # following step  (including way point (1.))
             [0. , 0.2, 0.4, 0.6, 0.8]]) # previous step  (excluding way point (2.))

    jac1 = ([[0.2, 0.4, 0.6, 0.8, 1. ],  # previous step  (including way point (2.))
             [0.8, 0.6, 0.4, 0.2, 0. ]])  # following step  (excluding way point (2.))

    """

    if order == 0:
        jac = (np.arange(n_substeps) / n_substeps)[np.newaxis, :].repeat(2, axis=0)
        jac[0, :] = 1 - jac[0, :]
    else:
        jac = (np.arange(start=n_substeps - 1, stop=-1, step=-1) / n_substeps)[np.newaxis, :].repeat(2, axis=0)
        jac[0, :] = 1 - jac[0, :]

    # jac /= n_substeps  # FIXME sum = 1, the influence of the substeps on the waypoints is independent from its numbers
    # Otherwise the center would always be 1, no matter how many substeps -> better this way
    # print('substeps normalization')
    return jac
