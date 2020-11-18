import numpy as np

from Kinematic import forward


def obstacle_collision_check(x_spheres, par):
    # Distance between spheres and environment
    oc_dist = par.dist_fun_r(x=x_spheres[:, :, par.active_spheres_idx, :])

    oc_dist = oc_dist.min(axis=-2)  # min over all time steps
    min_dist = oc_dist.min(axis=-1)  # min over all spheres

    feasible = np.array(min_dist > par.dist_threshold)

    return feasible


def obstacle_collision_check2(*, q, robot, par):
    x_spheres = forward.get_x_spheres_substeps(q=q, n_substeps=par.n_substeps_check, robot=robot)
    return obstacle_collision_check(x_spheres=x_spheres, par=par)


# Joint Limits
def box_constraints_limits_check(x, limits):
    n_samples, n_wp, n_dof = x.shape

    below_lower = x < limits[:, 0]
    above_upper = x > limits[:, 1]

    outside_limits = np.logical_or(below_lower, above_upper)
    outside_limits = outside_limits.reshape(n_samples, -1, n_dof)
    outside_limits = outside_limits.sum(axis=1)

    # Check the feasibility of each sample
    outside_limits = outside_limits.sum(axis=1)
    feasible = outside_limits == 0

    return feasible


def center_of_mass_check(*, frames, par, verbose=0):
    location_base = frames[:, :, par.base_frame_idx, :2, -1]
    location_com = frames[:, :, par.com_frame_idx, :2, -1]

    dist_com_norm = np.linalg.norm(location_com - location_base, axis=-1)
    max_dist_com = np.max(dist_com_norm, axis=-1)
    feasible = np.array(max_dist_com < par.dist_threshold)

    if verbose > 1:
        for i in range(np.size(feasible)):
            if not feasible[i] or verbose > 2:
                print('Maximal Center of Mass Distance: {:.4}m  -> Feasible: {}'.format(max_dist_com[i], feasible[i]))

    return feasible


def feasibility_check(*, q, par):
    n_samples = q.shape[0]

    frames, x_spheres = forward.get_x_spheres_substeps(q=q, robot=par.robot, n_substeps=par.oc.n_substeps_check,
                                                       return_frames2=True)

    # Obstacle Collision
    if par.check.obstacle_collision:
        feasible_oc = obstacle_collision_check(x_spheres=x_spheres, par=par.oc)
    else:
        feasible_oc = np.ones(n_samples, dtype=bool)

    # Joint Limits
    if par.check.limits:
        feasible_limits = box_constraints_limits_check(x=q, limits=par.robot.limits)
    else:
        feasible_limits = np.ones(n_samples, dtype=bool)

    # Override the status an return the smallest error value
    status = np.ones(n_samples, dtype=int)
    status[~feasible_limits] = -2
    status[~feasible_oc] = -1

    return status
