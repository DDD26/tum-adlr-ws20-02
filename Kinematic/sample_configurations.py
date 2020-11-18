import numpy as np
from wzk import random_uniform_ndim, mp_wrapper, get_n_samples_per_process


def __sample_q(*, robot, n_samples=1):
    if robot.id == 'Justin19':
        q = random_uniform_ndim(low=robot.world_limits[:, 0], high=robot.world_limits[:, 1], shape=(n_samples, 1))

        t2 = 1
        t3 = 2
        c_lower = np.full_like(q[..., t2], fill_value=robot.world_limits[t3, 0])
        c_upper = np.full_like(q[..., t2], fill_value=robot.world_limits[t3, 1])

        t2_below0 = q[..., t2] < 0
        c_lower[t2_below0] -= q[..., t2][t2_below0]
        c_upper[~t2_below0] -= q[..., t2][~t2_below0]
        q[..., t3] = c_lower + np.random.random((n_samples, 1)) * (c_upper - c_lower)
        return q
    else:
        return random_uniform_ndim(low=robot.limits[:, 0], high=robot.limits[:, 1], shape=(n_samples, 1))


def __sample_q_feasible(*, robot, feasibility_check, n_samples=1,
                        max_iter=20, verbose=0):
    count = 0
    safety_factor = 1.01

    q = np.zeros((n_samples, 1, robot.n_dof))
    feasible = None
    n_feasible = n_feasible_temp = 0
    n_samples_temp = n_samples // 2
    while n_feasible < n_samples:

        if n_feasible_temp == 0:
            n_samples_temp *= 2
        else:
            n_samples_temp = (feasible.size - feasible.sum()) / feasible.mean()
        n_samples_temp *= safety_factor
        n_samples_temp = max(int(np.ceil(n_samples_temp)), 1)

        q_temp = __sample_q(robot=robot, n_samples=n_samples_temp)
        feasible = feasibility_check(q=q_temp) >= 0
        n_feasible_temp = np.sum(feasible)

        q[n_feasible:n_feasible + n_feasible_temp, :, :] = q_temp[feasible, :, :][:n_samples - n_feasible, :, :]

        n_feasible += n_feasible_temp
        if verbose > 0:
            print('sample_valid_q', count, n_samples, n_feasible)
        if count >= max_iter:
            raise RuntimeError('Maximum number of iterations reached!')

        count += 1
    return q


def sample_q(*, robot, n_samples=1,
             feasibility_check=False, max_iter=20, verbose=0):
    if feasibility_check:
        bs = int(1e4)  # batch_size to compute the configurations sequentially, otherwise MemoryError

        if n_samples > bs:
            q = np.zeros((n_samples, 1, robot.n_dof))
            for i in range(n_samples // bs + 1):
                q[i * bs:(i + 1) * bs, :, :] = __sample_q_feasible(robot=robot,
                                                                   n_samples=int(min(bs, max(0, n_samples - i * bs))),
                                                                   feasibility_check=feasibility_check,
                                                                   max_iter=max_iter, verbose=verbose)
        else:
            q = __sample_q_feasible(robot=robot, n_samples=n_samples,
                                    feasibility_check=feasibility_check, max_iter=max_iter, verbose=verbose)

    else:
        q = __sample_q(robot=robot, n_samples=n_samples)  # without self-collision it is 250x faster

    return q


def sample_q_mp(*, robot, n_samples, feasibility_check=False, n_processes=1):
    """
    If valid is False is it faster for values < 1e5 to don't use multiprocessing,
    On galene there is a overhead og around 10ms for each process you start.
    """

    def fun_wrapper(n):
        return sample_q(robot=robot, n_samples=int(n), feasibility_check=feasibility_check)

    n_samples_per_core, _ = get_n_samples_per_process(n_samples=n_samples, n_processes=n_processes)
    return mp_wrapper(n_samples_per_core, fun=fun_wrapper, n_processes=n_processes)


def sample_q_frames_mp(*, robot, n_samples, feasibility_check=False, n_processes=1, frames_idx=None):
    if frames_idx is None:
        def fun(n):
            q = sample_q(robot=robot, n_samples=int(n), feasibility_check=feasibility_check)
            frames = robot.get_frames(q=q)
            return q, frames
    else:
        def fun(n):
            q = sample_q(robot=robot, n_samples=int(n), feasibility_check=feasibility_check)
            frames = robot.get_frames(q=q)[:, :, frames_idx, :, :]
            return q, frames

    n_samples_per_core, _ = get_n_samples_per_process(n_samples=n_samples, n_processes=n_processes)
    return mp_wrapper(n_samples_per_core, fun=fun, n_processes=n_processes)
