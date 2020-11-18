import numpy as np

import Kinematic.forward as forward
import Optimizer.objectives as obj
from Optimizer import feasibility_check, path


def get_feasible_optimum(*, q, par, q_close=None,
                         status=None, verbose=0,
                         return_cost=False):
    if status is None:
        status = feasibility_check(q=q, par=par)
    feasible = status >= 0

    if verbose > 0:
        print("{} of the {} solutions were feasible -> choose the best".format(np.sum(feasible), np.size(feasible)))
    if np.sum(feasible) == 0:
        status_unique, status_count = np.unique(status, return_counts=True)
        most_common_error = status_unique[np.argmax(status_count)]
        if return_cost:
            return q[0:0], most_common_error, np.inf
        else:
            return q[0:0], most_common_error

    q = q[feasible]

    if q_close is None:
        cost = obj.length_cost(q=q, joint_weighting=par.weighting.joint_motion,
                               infinity_joints=par.robot.infinity_joints)
    else:
        cost = obj.close_to_pose_cost(q=q, q_close=q_close, joint_weighting=par.weighting.joint_motion,
                                      infinity_joints=par.robot.infinity_joints)

    min_idx = np.argmin(cost)

    if return_cost:
        return q[min_idx:min_idx + 1, ...], 0, cost[min_idx]
    else:
        return q[min_idx:min_idx + 1, ...], 0


def get_optimal_goal_base(*, q_end, q_close, par):
    x_end, theta_end = path.q2x_q(xq=q_end, n_dim=par.size.n_dim)
    x_spheres_end = forward.get_x_spheres(q=q_end, robot=par.robot)

    # Obstacle-Collision
    cost_oc = par.oc.cost_fun(x=x_spheres_end[:, :, par.oc.active_spheres_idx, :])
    cost_oc = cost_oc.sum(axis=(1, 2)) / par.oc.active_spheres_idx.sum()
    feasible = cost_oc < 0.005

    print("Feasible Target positions: {} / {}".format(np.sum(feasible), np.size(feasible)))
    q_end = q_end[feasible]
    x_end = x_end[feasible]
    # q_close = q_close[feasible]

    cost_tcp_pos = np.linalg.norm(par.tcp.frame - x_end, axis=-1)

    cost_tcp_dist = np.abs(path.inf_joint_wrapper(x=q_close - q_end, inf_joints=par.robot.infinity_joints))
    cost_tcp_dist_x = np.linalg.norm(cost_tcp_dist[:, :, :2], axis=-1)
    cost_tcp_dist_theta = cost_tcp_dist[:, :, 2]

    cost = 1 * cost_tcp_dist_x + 15 * cost_tcp_dist_theta + 100 * cost_tcp_pos

    idx_min = np.argmin(cost)

    return q_end[idx_min], q_end[:, 0, :2]
