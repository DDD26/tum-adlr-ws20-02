import numpy as np

from Optimizer.path import get_x_substeps
from Kinematic import frames, chain as kc


def initialize_frames(shape, robot, mode='hm'):
    return frames.initialize_frames(shape=shape + (robot.n_frames,), n_dim=robot.n_dim, mode=mode)


def initialize_frames_jac(shape, robot, mode='hm'):
    f = initialize_frames(shape=shape, robot=robot, mode=mode)
    j = frames.initialize_frames(shape=shape + (robot.n_dof, robot.n_frames), n_dim=robot.n_dim, mode='zero')
    return f, j


# General
def get_frames(q, robot):
    return robot.get_frames(q)


def get_frames_jac(*, q, robot):
    return robot.get_frames_jacs(q=q)


def get_x_frames(*, q, robot):
    return robot.get_frames(q=q)[..., :-1, -1]


def frames2pos(f, frame_idx, rel_pos):
    return (f[:, :, frame_idx, :, :] @ rel_pos[:, :, np.newaxis])[..., :-1, 0]


def frames2spheres(f, robot):
    """
    x_spheres (n_samples, n_wp, n_links, n_dim)
    """
    return frames2pos(f, frame_idx=robot.spheres_frame_idx, rel_pos=robot.spheres_position)


def frames2spheres_jac(f, j, robot):
    """
    x_spheres (n_samples, n_wp, n_spheres, n_dim)
    dx_dq (n_samples, n_wp, n_dof, n_spheres, n_dim)
    """
    x_spheres = frames2spheres(f=f, robot=robot)
    dx_dq = (j[:, :, :, robot.spheres_frame_idx, :, :] @ robot.spheres_position[:, :, np.newaxis])[..., :-1, 0]

    return x_spheres, dx_dq


def get_x_spheres(q, robot, return_frames2=False):
    f = robot.get_frames(q=q)
    x_spheres = frames2spheres(f=f, robot=robot)
    if return_frames2:
        return f, x_spheres
    else:
        return x_spheres


def get_x_spheres_jac(*, q, robot, return_frames2=False):
    f, j = robot.get_frames_jac(q=q)
    x_spheres, dx_dq = frames2spheres_jac(f=f, j=j, robot=robot)
    if return_frames2:
        return (f, j), (x_spheres, dx_dq)
    else:
        return x_spheres, dx_dq


def get_x_spheres_substeps(*, q, robot, n_substeps, return_frames2=False):
    q_ss = get_x_substeps(x=q, n_substeps=n_substeps, infinity_joints=robot.infinity_joints, include_end_point=True)
    return get_x_spheres(q=q_ss, robot=robot, return_frames2=return_frames2)


def get_x_spheres_substeps_jac(*, q, robot, n_substeps, return_frames2=False):
    q_ss = get_x_substeps(x=q, n_substeps=n_substeps, infinity_joints=robot.infinity_joints, include_end_point=True)
    return get_x_spheres_jac(q=q_ss, robot=robot, return_frames2=return_frames2)


def get_frames_substeps(*, q, robot, n_substeps):
    q_ss = get_x_substeps(x=q, n_substeps=n_substeps, infinity_joints=robot.infinity_joints, include_end_point=True)
    return get_frames(q=q_ss, robot=robot)


def get_frames_substeps_jac(*, q, robot, n_substeps):
    q_ss = get_x_substeps(x=q, n_substeps=n_substeps, infinity_joints=robot.infinity_joints, include_end_point=True)
    return get_frames_jac(q=q_ss, robot=robot)


# nfi - next frame index
# iff - influence frame frame


# Helper
# Combine fun
def create_frames_dict(f, nfi):
    """
    Create a dict to minimize the calculation of unnecessary transformations between the frames

    The value to the key 0 holds all transformations form the origin to the whole chain.
    Each next field holds the transformation from the current frame to all frames to come.

    The calculation happens from back to front, to save some steps
    # 0     1     2     3     4
    # F01
    # F02   F12
    # F03   F13   F23
    # F04   F14   F24   F34
    # F05   F15   F25   F35   F45

    """
    n_frames = f.shape[-3]

    d = {}
    for i in range(n_frames - 1, -1, -1):
        nfi_i = nfi[i]

        if nfi_i == -1:
            d[i] = f[..., i:i + 1, :, :]

        elif isinstance(nfi_i, (list, tuple)):
            d[i] = np.concatenate([
                f[..., i:i + 1, :, :],
                f[..., i:i + 1, :, :] @ np.concatenate([d[j] for j in nfi_i], axis=-3)],
                axis=-3)

        else:
            d[i] = np.concatenate([f[..., i:i + 1, :, :],
                                   f[..., i:i + 1, :, :] @ d[nfi_i]], axis=-3)
    return d


def combine_frames(f, prev_frame_idx):
    for i, pfi in enumerate(prev_frame_idx[1:], start=1):
        f[..., i, :, :] = f[..., pfi, :, :] @ f[..., i, :, :]


def combine_frames_jac(j, d, robot):
    jf_all, jf_first, jf_last = kc.__get_joint_frame_indices_first_last(jfi=robot.joint_frame_idx)

    pfi_ = robot.prev_frame_idx[jf_first]
    joints_ = np.arange(robot.n_dof)[pfi_ != -1]
    jf_first_ = jf_first[pfi_ != -1]
    pfi_ = pfi_[pfi_ != -1]

    # Previous to joint frame
    # j(b)__a_b = f__a_b * j__b
    j[..., joints_, jf_first_, :, :] = (d[0][..., pfi_, :, :] @ j[..., joints_, jf_first_, :, :])

    # After
    for i in range(robot.n_dof):
        jf_inf_i = robot.joint_frame_influence[i, :]
        jf_inf_i[:jf_last[i] + 1] = False
        nfi_i = robot.next_frame_idx[jf_last[i]]

        # Handle joints which act on multiple frames
        if jf_first[i] != jf_last[i]:
            for kk, fj_cur in enumerate(jf_all[i][:-1]):
                jf_next = jf_all[i][kk + 1]
                jf_next1 = jf_next - 1

                if jf_next - fj_cur > 1:
                    j[..., i, fj_cur + 1:jf_next, :, :] = (j[..., i, fj_cur:fj_cur + 1, :, :] @
                                                           d[robot.next_frame_idx[fj_cur]][..., :jf_next - fj_cur - 1, :, :])

                j[..., i, jf_next, :, :] = ((j[..., i, jf_next1, :, :] @ d[robot.next_frame_idx[jf_next1]][..., 0, :, :]) +
                                            (d[0][..., jf_next1, :, :] @ j[..., i, jf_next, :, :]))

        # j(b)__a_c = j__a_b * f__b_c
        if isinstance(nfi_i, (list, tuple)):
            j[..., i, jf_inf_i, :, :] = (j[..., i, jf_last[i]:jf_last[i] + 1, :, ] @ np.concatenate([d[j] for j in nfi_i], axis=-3))
        elif nfi_i != -1:
            j[..., i, jf_inf_i, :, :] = (j[..., i, jf_last[i]:jf_last[i] + 1, :, :] @ d[nfi_i])
