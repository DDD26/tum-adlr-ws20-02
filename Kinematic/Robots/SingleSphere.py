import numpy as np
from Kinematic.forward import initialize_frames, initialize_frames_jac
from Kinematic.frames import fill_frames_trans_add
from Kinematic.Robots import Robot


class SingleSphere(Robot):
    def __init__(self, n_dim, radius=0.3):
        self.id = f'SingleSphere0{n_dim}'
        self.n_dim = n_dim
        self.n_dof = self.n_dim

        self.f_world_robot = None
        self.infinity_joints = np.zeros(self.n_dof, dtype=bool)
        self.limits = None

        self.spheres_position = np.zeros((1, self.n_dim + 1))
        self.spheres_position[:, -1] = 1
        self.spheres_radius = np.full(1, fill_value=radius)
        self.spheres_frame_idx = np.zeros(1, dtype=int)

        self.n_frames = 1
        # self.next_frame_idx = np.array([-1])
        # self.prev_frame_idx = np.array([-1])
        # self.joint_frame_idx = np.zeros((0,))
        # self.joint_frame_influence = np.ones((0, 1))

    def get_frames(self, q):
        f = initialize_frames(shape=q.shape[:-1], robot=self, mode='eye')
        fill_frames_trans_add(f=f, trans=q[..., np.newaxis, :])
        return f

    def get_frames_jac(self, q):
        f, j = initialize_frames_jac(shape=q.shape[:-1], robot=self, mode='eye')
        fill_frames_trans_add(f=f, trans=q[..., np.newaxis, :])
        fill_frames_jac__dx(j=j, n_dim=self.n_dim)
        return f, j


class SingleSphere02(SingleSphere):
    def __init__(self, radius):
        super().__init__(n_dim=2, radius=radius)


class SingleSphere03(SingleSphere):
    def __init__(self, radius):
        super().__init__(n_dim=3, radius=radius)


def fill_frames_jac__dx(j, n_dim):
    """
    Assume that the dof xy(z) are the first 2(3)
    """
    for i in range(n_dim):
        j[:, :, i, :, i, -1] = 1