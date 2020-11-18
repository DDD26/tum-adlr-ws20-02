import numpy as np
from Kinematic import forward, frames

from Kinematic.Robots.SingleSphere import fill_frames_jac__dx
from Kinematic.Robots.StaticArm import fill_frames_arm2d, fill_frames_jac_arm2d, get_arm2d_spheres, create_serial_robot

from wzk import safe_scalar2array


class MovingArm:
    def __init__(self, n_dof, limb_lengths=0.5, radius=0.1):
        self.id = f'StaticArm{n_dof}'
        self.n_dim = 2
        self.n_dof_arm = n_dof
        self.n_dof = self.n_dim + self.n_dof_arm

        self.f_world_robot = None
        self.infinity_joints = np.ones(self.n_dof, dtype=bool)
        self.infinity_joints[:self.n_dim] = False

        self.limits = np.repeat(np.array([[-np.pi, np.pi]]), self.n_dof, axis=0)
        self.limits[:self.n_dim, :] = np.array([[0, 10],
                                                [0, 10]])

        self.limb_lengths = limb_lengths

        self.spheres_position, self.spheres_frame_idx = \
            get_arm2d_spheres(n_links=self.n_dof_arm, spheres_per_link=3, limb_lengths=self.limb_lengths)

        self.spheres_radius = safe_scalar2array(0.1, shape=len(self.spheres_frame_idx))

        (self.next_frame_idx, self.prev_frame_idx, self.joint_frame_idx,
         self.frame_frame_influence, self.joint_frame_influence) = create_serial_robot(n_dof=self.n_dof)

    def get_frames(self, q):
        f = forward.initialize_frames(shape=q.shape[-1], robot=self, mode='zero')
        x, q = q[:, :, :self.n_dim], q[:, :, self.n_dim:]
        sin, cos = np.sin(q), np.cos(q)
        fill_frames_arm2d(sin=sin, cos=cos, f=f, limb_lengths=self.limb_lengths)
        forward.combine_frames(f=f, prev_frame_idx=self.prev_frame_idx)
        frames.fill_frames_trans_add(f=f, trans=x[:, :, np.newaxis, :])
        f = frames.apply_eye_wrapper(f=f, possible_eye=self.f_world_robot)
        return f

    def get_frames_jac(self, q):
        f, j = forward.initialize_frames_jac(shape=q.shape[-1], robot=self, mode='zero')
        x, q = q[:, :, :self.n_dim], q[:, :, self.n_dim:]
        sin, cos = np.sin(q), np.cos(q)
        fill_frames_jac_arm2d(sin=sin, cos=cos, f=f, j=j[..., self.n_dim:, :, :, :],
                              joint_frame_idx=self.joint_frame_idx, limb_lengths=self.limb_lengths)

        d = forward.create_frames_dict(f=f, nfi=self.next_frame_idx)

        forward.combine_frames_jac(j=j[..., self.n_dim:, :, :, :], d=d, robot=self)

        f = d[0]

        frames.fill_frames_trans_add(f=f, trans=x[..., np.newaxis, :])
        fill_frames_jac__dx(j=j, n_dim=self.n_dim)

        f = frames.apply_eye_wrapper(f=f, possible_eye=self.f_world_robot)
        return f, j


class MovingTree:
    pass