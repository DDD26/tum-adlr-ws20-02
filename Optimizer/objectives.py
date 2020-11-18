import numpy as np

import Kinematic.forward as forward
import Optimizer.path as path


def jac_combine_substeps(jac_ss, n_substeps,
                         n_samples, n_joints):
    # Combine the jacobians of the sub-way-points (joints) to the jacobian for the optimization variables
    if n_substeps > 1:
        jac_ss = jac_ss.reshape(n_samples, -1, n_substeps, n_joints)
        ss_jac = path.d_substeps__dx(n_substeps=n_substeps, order=1)[np.newaxis, np.newaxis, ..., np.newaxis]
        jac = np.einsum('ijkl, ijkl -> ijl', jac_ss, ss_jac[:, :, 0, :, :])
        jac[:, :-1, :] += np.einsum('ijkl, ijkl -> ijl', jac_ss[:, 1:, :, :], ss_jac[:, :, 1, :, :])
        return jac
    else:
        return jac_ss


def length_cost(q, infinity_joints, joint_weighting=1):
    """
    Calculate the squared euclidean distance of each step and sum up over the path and over the joints.
    Consider infinity joints for the step shape calculation and make a weighting between the different joints
    Because the squared step shape is considered it is not necessary to handle cartesian coordinates x,y,z
    differently from the joints.
    """

    q_steps = np.diff(q, axis=-2)
    q_steps = path.inf_joint_wrapper(x=q_steps, inf_joints=infinity_joints)
    len_cost = (q_steps ** 2).sum(axis=-2)  # Sum over path of the quadratic step length for each dof

    # Weighted sum over the joints
    len_cost = 0.5 * (len_cost * joint_weighting).sum(axis=-1)

    return len_cost


def length_grad(q, infinity_joints, joint_weighting, jac_only=True):
    """
    Jacobian for 'trajectory_length(x, squared=True)' with respect to the coordinate (j) of
    a way point (o).
    length_squared = ls = 1/2 * Sum(o=1, n) |p_i - p_(o-1)|**2
    # Start- and endpoint: x 1
    # Points in the middle, with two neighbours: x 2
    """

    q_steps = path.get_x_steps(x=q)
    q_steps = path.inf_joint_wrapper(q_steps, inf_joints=infinity_joints)

    jac = q_steps[..., :-1, :] - q_steps[..., +1:, :]

    jac *= joint_weighting  # Broadcasting starts from trailing dimension (numpy broadcasting)

    if jac_only:
        return jac
    else:
        len_cost = (q_steps ** 2).sum(axis=-2)
        len_cost = 0.5 * (len_cost * joint_weighting).sum(axis=-1)
        # raise NotImplementedError
        # If end position of optimization the length norm can not be computed beforehand
        # len_cost = length_cost_a(x=x, a=a,
        #                          n_joints=n_joints, fixed_base=fixed_base, infinity_joints=infinity_joints,
        #                          length_a_penalty=length_a_penalty, length_norm=length_norm,
        #                          n_dim=n_dim, n_samples=n_samples)
        return len_cost, jac


def obstacle_collision_cost(x_spheres, oc):
    """
    Approximate the obstacle cost of a trajectory by summing up the values at substeps and weight them by the length of
    these steps. Approximation converges to integral for n_substeps -> inf

    Either define the number of substeps directly ('n_substeps',  5 is a good guess) or use a relative metric
    ('substep_ratio'). Here the largest step shape of the path divided by the cell shape of the grid is chosen
    number of substeps (times safety factor).

    Start point is missing but there the edt should be zero anyway (start  is not inside an obstacle)
    """

    x_spheres = x_spheres[:, :, oc.active_spheres_idx, :]
    _, n_wp_ss, n_spheres, _ = x_spheres.shape

    # Get substep lengths
    if n_wp_ss == 1:
        x_spheres_step_length = np.ones((1, 1, 1))
    else:
        x_spheres_step_length = np.linalg.norm(np.diff(x_spheres, axis=1), axis=-1)
        x_spheres = x_spheres[:, 1:, :, :]  # Ignore the start point

    # Obstacle distance cost at the substeps
    spheres_obst_cost = oc.cost_fun(x=x_spheres)

    # Sum over the sub steps, Mean over the spheres
    obst_cost = np.einsum("ijk, ijk -> i", spheres_obst_cost, x_spheres_step_length) / n_spheres

    return obst_cost


def obstacle_collision_grad(x_spheres, dx_dq,
                            oc,
                            jac_only=True):
    x_spheres = x_spheres[:, :, oc.active_spheres_idx, :]
    dx_dq = dx_dq[:, :, :, oc.active_spheres_idx, :]

    n_samples, n_wp_ss, n_joints, n_spheres, n_dim = dx_dq.shape

    if n_wp_ss == 1:
        oc_cost, oc_cost_derv = oc.cost_fun_derv(x_spheres)
        jac = (dx_dq * oc_cost_derv[:, :, np.newaxis, :, :]).sum(axis=(-2, -1)) / n_spheres
        if jac_only:
            return jac
        else:
            oc_cost = oc_cost.sum(axis=(-2, -1)) / n_spheres
            return oc_cost, jac

    # Calculate different parts for the jacobians
    # (Sub) Step length + Jac
    x_spheres_steps = np.diff(x_spheres, axis=1)
    x_spheres_step_length = np.linalg.norm(x_spheres_steps, axis=-1, keepdims=True)
    ssl_jac = path.d_step_length__dx(steps=x_spheres_steps, step_lengths=x_spheres_step_length[..., 0])  # sub step le

    # EDT cost and derivative at the substeps
    x_spheres = x_spheres[:, 1:, :, :]  # Ignore starting point, should be by definition collision free
    x_ss_cost, x_ss_cost_derv = oc.cost_fun_derv(x=x_spheres)
    x_ss_cost = x_ss_cost[..., np.newaxis]

    # Combine the expressions to get the final jacobian
    #              v     *   l'
    jac = x_ss_cost * ssl_jac
    jac[:, :-1, :, :] -= jac[:, 1:, :, :]  # (Two contributions, from left and right neighbour)

    #                v'        *        l
    jac += x_ss_cost_derv * x_spheres_step_length

    # Sum over (x, y, z and the spheres (mean))
    jac = np.einsum("ijklm, ijklm->ijk", jac[:, :, np.newaxis, :, :], dx_dq) / n_spheres

    # Combine the jacobians of the sub-way-points (joints) to the jacobian for the optimization variables
    jac = jac_combine_substeps(jac_ss=jac, n_substeps=oc.n_substeps_derv, n_samples=n_samples, n_joints=n_joints)
    jac = jac[:, :-1, :]

    if jac_only:
        return jac
    else:
        obst_cost = x_ss_cost[..., 0] * x_spheres_step_length[..., 0]
        obst_cost = obst_cost.sum(axis=(-2, -1)) / n_spheres  # TODO mean
        return obst_cost, jac


def chomp_cost(q, par, return_separate=False):
    x_spheres = forward.get_x_spheres_substeps(q=q, n_substeps=par.oc.n_substeps_cost, robot=par.robot)

    # Obstacle Cost
    obst_cost = obstacle_collision_cost(x_spheres=x_spheres, oc=par.oc)

    # Length Cost
    len_cost = length_cost(q=q, joint_weighting=par.weighting.joint_motion, infinity_joints=par.robot.infinity_joints)

    if return_separate:
        return len_cost, obst_cost
    else:
        return __perform_weighting(weighting=par.weighting, oc=obst_cost, dist=len_cost)


def chomp_grad(q, par,
               jac_only=True, return_separate=False):
    x_spheres, dx_dq = forward.get_x_spheres_substeps_jac(q=q, robot=par.robot, n_substeps=par.oc.n_substeps_derv)
    dx_dq = dx_dq[:, 1:, :, :, :]  # ignore start point, should be feasible anyway

    # Obstacle Cost
    obst_jac = obstacle_collision_grad(x_spheres=x_spheres, dx_dq=dx_dq, oc=par.oc,
                                       jac_only=jac_only)

    # Length Cost
    len_jac = length_grad(q=q,
                          joint_weighting=par.weighting.joint_motion,
                          infinity_joints=par.robot.infinity_joints, jac_only=jac_only)

    if not jac_only:
        len_cost, len_jac = len_jac
        obst_cost, obst_jac = obst_jac
    else:
        len_cost = obst_cost = None

    if return_separate:
        jac = len_jac, obst_jac
        if jac_only:
            return jac
        else:
            _cost = len_cost, obst_cost
            return _cost, jac
    else:
        jac = __perform_weighting(weighting=par.weighting, oc=obst_jac, dist=len_jac)

        if jac_only:
            return jac
        else:
            _cost = __perform_weighting(weighting=par.weighting, oc=obst_cost, dist=len_cost)
            return _cost, jac


def close_to_pose_cost(q, q_close, joint_weighting, infinity_joints):
    cost_close2pose = path.inf_joint_wrapper(x=q_close - q, inf_joints=infinity_joints) ** 2
    cost_close2pose = 0.5 * (cost_close2pose * joint_weighting).sum(axis=(-2, -1))
    return cost_close2pose


def close_to_pose_grad(q, q_close, joint_weighting, infinity_joints):
    grad_close2pose = - path.inf_joint_wrapper(x=q_close - q, inf_joints=infinity_joints) * joint_weighting
    return grad_close2pose


def __perform_weighting(weighting, oc=0, dist=0):
    return weighting.collision * oc + weighting.length * dist
