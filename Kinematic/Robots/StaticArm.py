import numpy as np
from Kinematic import forward, frames, chain

from wzk import safe_scalar2array


# TODO old stat arm
# print('old arm')
# robot.limb_lengths = 5.19615242 * 4 / 10
# robot.spheres_position, robot.spheres_frame_idx = \
#     __get_arm2d_spheres(n_links=robot.n_dof, spheres_per_link=4, limb_lengths=robot.limb_lengths)

# robot.limb_lengths = 5.19615242 * 3 / 10
# robot.spheres_position, robot.spheres_frame_idx = \
#     __get_arm2d_spheres(n_links=robot.n_dof, spheres_per_link=3, limb_lengths=robot.limb_lengths)

def get_world_limits(n_dof, limb_lengths):
    world_limits = np.empty((2, 2))
    world_limits[:, 0] = -(n_dof + 0.5) * limb_lengths
    world_limits[:, 1] = +(n_dof + 0.5) * limb_lengths
    return world_limits


def get_arm2d_spheres(n_links, spheres_per_link, limb_lengths):
    """radius * 2 * np.cos(np.arcsin(1 / 2))"""
    n_dim = 2
    n_dim1 = n_dim+1

    spheres_per_link, limb_lengths = safe_scalar2array(spheres_per_link, limb_lengths, shape=n_links)

    n_spheres = np.sum(spheres_per_link) + 1
    spheres_position = np.zeros((n_spheres, n_dim1))
    spheres_position[:, -1] = 1
    spheres_position[1:, 0] = np.concatenate([np.linspace(0, ll, spl+1)[1:] for ll, spl
                                              in zip(limb_lengths, spheres_per_link)])

    spheres_frame_idx = np.zeros(n_spheres, dtype=int)
    spheres_frame_idx[1:] = np.concatenate([np.full(spl, i) for i, spl in enumerate(spheres_per_link)])

    return spheres_position, spheres_frame_idx


class StaticArm:
    def __init__(self, n_dof, limb_lengths=0.5, radius=0.1):
        self.id = f'StaticArm{n_dof}'
        self.n_dim = 2
        self.n_dof = n_dof

        self.f_world_robot = None
        self.infinity_joints = np.ones(self.n_dof, dtype=bool)
        self.limits = np.repeat(np.array([[-np.pi, np.pi]]), self.n_dof, axis=0)

        self.limb_lengths = limb_lengths
        self.spheres_position, self.spheres_frame_idx = \
            get_arm2d_spheres(n_links=self.n_dof, spheres_per_link=3, limb_lengths=self.limb_lengths)
        self.spheres_radius = safe_scalar2array(radius, shape=len(self.spheres_frame_idx))

        (self.next_frame_idx, self.prev_frame_idx, self.joint_frame_idx,
         self.frame_frame_influence, self.joint_frame_influence) = create_serial_robot(n_dof=self.n_dof)
        self.n_frames = len(self.next_frame_idx)

    def get_frames(self, q):
        f = forward.initialize_frames(shape=q.shape[-1], robot=self, mode='zeros')
        sin, cos = np.sin(q), np.cos(q)
        fill_frames_arm2d(sin=sin, cos=cos, f=f, limb_lengths=self.limb_lengths)
        forward.combine_frames(f=f, prev_frame_idx=self.prev_frame_idx)
        return f

    def get_frames_jac(self, q):
        f, j = forward.initialize_frames_jac(shape=q.shape[-1], robot=self, mode='zeros')

        sin, cos = np.sin(q), np.cos(q)
        fill_frames_jac_arm2d(sin=sin, cos=cos, f=f, j=j,
                              joint_frame_idx=self.joint_frame_idx, limb_lengths=self.limb_lengths)

        dh_dict = forward.create_frames_dict(f=f, nfi=self.next_frame_idx)
        forward.combine_frames_jac(j=j, d=dh_dict, robot=self)
        f = dh_dict[0]
        f = frames.apply_eye_wrapper(f=f, possible_eye=self.f_world_robot)
        return f, j


class StaticTree:
    # TODO make this easier usable, make it as close to par as possible
    def __init__(self):
        self.next_frame_idx = [1, [2, 4], 3, -1, 5, -1]
        self.prev_frame_idx = chain.next2prev_frame_idx(nfi=self.next_frame_idx)
        self.joint_frame_idx = np.arange(len(self.next_frame_idx))
        self.joint_frame_influence = chain.influence_frames_frames2joints_frames(jfi=self.joint_frame_idx,
                                                                                 nfi=self.next_frame_idx)

        self.n_dim = 2
        self.n_dof = len(self.next_frame_idx)

        self.f_world_robot = None

        self.limb_lengths = 0.5
        self.infinity_joints = np.ones(self.n_dof, dtype=bool)
        self.limits = np.repeat(np.array([[-np.pi, np.pi]]), self.n_dof, axis=0)

        self.spheres_position, self.spheres_frame_idx = \
            get_arm2d_spheres(n_links=self.n_dof, spheres_per_link=3, limb_lengths=self.limb_lengths)

        self.spheres_radius = safe_scalar2array(0.1, shape=len(self.spheres_frame_idx))


def fill_frames_jac_arm2d(sin, cos, f, j, joint_frame_idx, limb_lengths):
    fill_frames_arm2d(sin=sin, cos=cos, f=f, limb_lengths=limb_lengths)
    fill_jacs_arm2d(sin=sin, cos=cos, j=j, joint_frame_idx=joint_frame_idx)


def fill_frames_arm2d(sin, cos, f, limb_lengths):
    frames.fill_frames_2d_sc(sin=sin, cos=cos, f=f[..., :-1, :, :])
    frames.fill_frames_2d_xy(x=limb_lengths, f=f[..., 1:, :, :])
    frames.fill_frames_diag(f[..., -1, :, :])


def fill_jacs_arm2d(*, sin, cos, j, joint_frame_idx):
    joint_diag = range(j.shape[-4])
    frames.fill_frames_jac_2d_sc(j=j[:, :, joint_diag, joint_frame_idx, :, :], sin=sin, cos=cos)


def create_serial_robot(n_dof):
    """
    Include a TCP after the last frame
    """
    nfi = np.arange(1, n_dof + 2)
    nfi[-1] = -1
    jf_idx = np.arange(n_dof)
    pfi = chain.next2prev_frame_idx(nfi=nfi)
    ff_inf = chain.next_frame_idx2influence_frames_frames(nfi=nfi)
    jf_inf = chain.influence_frames_frames2joints_frames(nfi=nfi, jfi=jf_idx)
    return nfi, pfi, jf_idx, ff_inf, jf_inf
