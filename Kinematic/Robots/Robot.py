import numpy as np
from Optimizer.path import inf_joint_wrapper


class Robot(object):
    """
    Parameter describing the robot
    """
    __slots__ = ('id',                     # str                       | Unique name of the robot
                 'n_dof',                  # int                       | Degrees of freedom
                 'n_dim',                  # int                       | Number of spacial dimensions, 2D/3D
                 'limits',                 # float[n_dof][2]           | (min, max) value for each dof
                 'infinity_joints',        # bool[n_dof]               | Array indicating which joints are limitless
                 'f_world_robot',          # float[n_dim+1][n_dim+1]   | Position base frame in the world

                 'spheres_radius',         # float[n_spheres]          |
                 'spheres_frame_idx',      # int[n_spheres]            | The frame in which each sphere is fixed
                 'spheres_position',       # float[n_spheres][n_dim+1] | The homogeneous coordinates of each sphere
                 #                                                     | with respect to its frame
                 'limb_lengths',           # float[n_frames-1]         | Length of each limb [m]

                 'n_frames',               # int                       |
                 'next_frame_idx',         # int[n_frames][?]          | List of Lists indicating the next frame(s)
                 'prev_frame_idx',         # int[n_frames]             | Array indicating the previous frame
                 'joint_frame_idx',        # int[n_frames][?]          | List of Lists showing the leverage point
                 #                                                     | of each joint, (only necessary if coupled)
                 'joint_frame_influence',  # bool[n_dof][n_frames]     | Matrix showing the influence of each joint
                 'frame_frame_influence',  # bool[n_frames][n_frames]  | Matrix showing the influence of each frame
                 )

    def prune_joints2limits(self, q):
        q = inf_joint_wrapper(q, inf_joints=self.infinity_joints)
        return np.clip(q, a_min=self.limits[:, 0], a_max=self.limits[:, 1])

    def get_frames(self, q):
        pass

    def get_frames_jac(self, q):
        pass
