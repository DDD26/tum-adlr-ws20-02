import copy
import numpy as np

from wzk import safe_scalar2array, safe_unify


class CopyableObject(object):
    __slots__ = ()

    def copy(self):
        return copy.copy(self)


class GradientDescent(CopyableObject):
    __slots__ = ('n_steps',              # int                 | Number of iterations
                 'step_size',            # float                |
                 'adjust_step_size',     # bool                |
                 'staircase_iteration',  # bool                |
                 'grad_tol',             # float[n_steps]       |
                 'callback',             # fun()               |
                 'prune_limits',         # fun()               |
                 'n_processes',          # int                 |
                 'hesse_inv',            # float[n_var][n_var]  |
                 'hesse_weighting',      # float[n_steps]       |
                 'return_x_list'         # bool                |  is this a suitable parameter? not really
                 )

    def __init__(self):
        self.n_steps = 10
        self.step_size = 0.001
        self.adjust_step_size = True
        self.staircase_iteration = False
        self.grad_tol = 1

        self.n_processes = 1
        self.callback = None
        self.prune_limits = None
        self.hesse_inv = None
        self.hesse_weighting = 0

        self.return_x_list = False


class ProblemSize(object):
    __slots__ = ('n_dim',        # int |
                 'n_dof',        # int |
                 'n_samples',    # int |
                 'n_waypoints',  # int |
                 'n_substeps'    # int |
                 )


class World(object):
    __slots__ = ('n_dim',           # int             | Number of spacial dimensions, 2D/3D
                 'limits',          # float[n_dim][2] | Lower and upper boundary of each spatial dimension [m]
                 'size',            # float[n_dim]    | Size of the world in each dimension in meters
                 'n_voxels',        # int[n_dim]      | Size of the world in each dimension in voxels
                 'voxel_size',      # float           | Side length of one pixel/voxel [m]
                                    #                 | (depends on limits and n_voxels, make sure all voxels are cubes)
                 )

    def __init__(self, n_dim, limits, n_voxels=None, voxel_size=None):

        self.n_dim = n_dim

        if limits is None:
            self.limits = np.zeros((n_dim, 2))
            self.limits[:, 1] = 10
        else:
            self.limits = limits

        self.size = np.diff(self.limits, axis=-1)[:, 0]

        if n_voxels is None:
            self.n_voxels = (self.size / self.voxel_size).astype(int)
        else:
            self.n_voxels = safe_scalar2array(n_voxels, shape=self.n_dim)

        if voxel_size is None:
            self.voxel_size = safe_unify(self.size / n_voxels)


class ObstacleCollision(object):
    __slots__ = (
        'active_spheres_idx',      # bool[n_spheres]  | Indicates which spheres are active for collision
        'n_substeps_cost',         # int              | Number of substeps used in the cost function
        'n_substeps_derv',         # int              | Number of substeps used in its derivative
        'n_substeps_check',        # int              | Number of substeps used in its derivative
        'dist_fun_r',              # fun()            |
        'dist_derv',               # fun()            |
        'cost_fun',                # fun()            |
        'cost_fun_derv',           # fun()            |
        'edt_interp_order_cost',   # int              | Interp. order for extracting the values from the edt(0)
        'edt_interp_order_derv',   # int              | ... from the spacial derivative of the edt (1)
        'eps_dist_cost',           # float            | additional safety length for which the cost is smoothed
        #                                             | out quadratically [m] (0.05)
        'dist_threshold'           # float            |
        )


class CheckingType(object):
    __slots__ = ('obstacle_collision',  # bool |
                 'limits',              # bool |
                 )


class Weighting(CopyableObject):
    __slots__ = ('length',                        # float[gd.n_steps] |
                 'collision',                     # float[gd.n_steps] |
                 'joint_motion',                  # float[shape.n_dof] |
                 )

    def at_idx(self, i):
        # TODO cleaner
        new_weighting = self.copy()
        if np.size(new_weighting.length) > 1:
            new_weighting.length = new_weighting.length[i]
        if np.size(new_weighting.collision) > 1:
            new_weighting.collision = new_weighting.collision[i]

        return new_weighting

    def at_range(self, start, stop):
        new_weighting = self.copy()
        if np.size(new_weighting.length) > 1:
            new_weighting.length = new_weighting.length[start:stop]
        if np.size(new_weighting.collision) > 1:
            new_weighting.collision = new_weighting.collision[start:stop]
        return new_weighting


class Parameter(object):
    __slots__ = (
                 'robot',       # Robot, Kinematic + Sphere Model
                 'world',       # World, Limits + Voxels
                 'size',        # Problem dimensions
                 'oc',          # Obstacle Collision
                 'sc',          # Self-Collision
                 'com',         # Center of Mass
                 'tcp',         # Tool Center Point
                 'pbp',         # Pass by Points
                 'planning',    # Planning options
                 'weighting',   # Weighting factors between the different parts of the cost-function
                 'check',       #
    )       #


def initialize_oc_par(*, obstacle_img, oc, world, robot):
    import GridWorld.obstacle_distance as dist_f

    try:
        dummy = oc.active_spheres_idx
    except AttributeError:
        oc.active_spheres_idx = np.ones_like(robot.spheres_radius, dtype=bool)

    oc.dist_fun_r, oc.dist_derv, oc.cost_fun, oc.cost_fun_derv = \
        dist_f.obstacle_img2functions(obstacle_img=obstacle_img, add_boundary=True,
                                      voxel_size=world.voxel_size, lower_left=world.limits[:, 0],
                                      spheres_radius=robot.spheres_radius[oc.active_spheres_idx],
                                      interp_order_dist=oc.edt_interp_order_cost,
                                      interp_order_grad=oc.edt_interp_order_derv,
                                      eps=oc.eps_dist_cost)

    return obstacle_img


def initialize_par() -> object:
    par = Parameter()

    par.size = ProblemSize()
    par.size.n_samples = 1

    par.weighting = Weighting()
    par.weighting.joint_motion = 1

    par.oc = ObstacleCollision()
    par.oc.edt_interp_order_cost = 1
    par.oc.edt_interp_order_derv = 0
    par.oc.n_substeps_cost = 1
    par.oc.n_substeps_derv = 1
    par.oc.n_substeps_check = 1
    par.oc.eps_dist_cost = 0.025
    par.oc.dist_threshold = -0.005

    par.check = CheckingType()
    par.check.limits = True
    par.check.obstacle_collision = True

    return par
