import numpy as np

from Kinematic.Robots import SingleSphere02
from GridWorld.random_rectangles import create_rectangle_image
from Optimizer import path, choose_optimum, initial_guess, gradient_descent, objectives
import plotting
import parameter

# Sample Program for showing the Optimizer Based Motion Planning (OMP) as described by CHOMP
# Most of the parameters

np.random.seed(2)
par = parameter.initialize_par()


world_limits = np.array([[0, 10],   # x [mn]
                         [0, 10]])  # y [m]
n_voxels = 64
radius = 0.3  # Size of the robot [m]

par.robot = SingleSphere02(radius=radius)
par.robot.limits = world_limits
par.world = parameter.World(n_dim=2, n_voxels=n_voxels, limits=world_limits)

par.size.n_waypoints = 20  # number of the discrete way points of the trajectory
par.oc.n_substeps_cost = 3  # number of sub steps to checl for obstacle collision
par.oc.n_substeps_derv = 3

# Gradient Descent
gd = parameter.GradientDescent()
gd.n_steps = 50
gd.step_size = 1
gd.adjust_step_size = True
gd.grad_tol = np.full(gd.n_steps, 1)
gd.n_processes = 1

# Weighting of Objective Terms, it is possible that each gradient step as a different weighting between those terms
par.weighting = parameter.Weighting()
par.weighting.joint_motion = np.array([1, 1])
par.weighting.length = np.linspace(start=1, stop=1, num=gd.n_steps)
par.weighting.collision = np.linspace(start=10, stop=100, num=gd.n_steps)

# Number of multi starts
n_multi_start_rp = [[0, 1, 2],  # how many random points for the multi-start (0 == straight line)
                    [1, 5, 4]]  # how many trials for each variation
get_q0 = initial_guess.q0_random_wrapper(robot=par.robot, n_multi_start=n_multi_start_rp,
                                         n_waypoints=par.size.n_waypoints, order_random=True, mode='inner')


# Create random obstacle image + functions for the Signed Distance Field
n_obstacles = 10
min_max_obstacle_size_voxel = [1, 8]
obstacle_img = create_rectangle_image(n=n_obstacles, size_limits=min_max_obstacle_size_voxel, n_voxels=par.world.n_voxels)
parameter.initialize_oc_par(oc=par.oc, world=par.world, robot=par.robot, obstacle_img=obstacle_img)


# Choose sample start and goal point of the motion problem
q_start, q_end = np.array([[[1, 1]]]), np.array([[[9, 9]]])

# Get initial guesses for given start and end
q0 = get_q0(start=q_start, end=q_end)




#####
# Perform Gradient Descent
x_initial, q_ms, objective = gradient_descent.solve_chomp_ms(q0=q0, q_start=q_start, q_end=q_end,
                                                             gd=gd, par=par, return_all=True, verbose=1)




# Choose the optimal trajectory from all multi starts
q_opt, _ = choose_optimum.get_feasible_optimum(q=q_ms, par=par, verbose=2)
q_opt = path.x_inner2x(x_inner=q_opt, x_start=q_start, x_end=q_end)


# Plot multi starts and optimal solution
fig, ax = plotting.new_world_fig(limits=par.world.limits, title='Multi-Starts')
plotting.plot_img_patch_w_outlines(img=obstacle_img, limits=par.world.limits, ax=ax)
for q in q_ms:
    plotting.plot_x_path(x=q, r=par.robot.spheres_radius, ax=ax, marker='o',)

plotting.plot_x_path(x=q_opt, r=par.robot.spheres_radius, ax=ax, marker='o', color='k')


#
cost, (length_jac, collision_jac) = objectives.chomp_grad(q=q_ms, par=par, jac_only=False, return_separate=True)
