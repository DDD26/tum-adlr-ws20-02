import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import torch
import gym
from gym import spaces
from stable_baselines import PPO2, TD3, A2C
# from stable_baselines3 import PPO, A2C, SAC  # DQN coming soon
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

"""Problems: 1. 1/eps
            - 2. weight = Default 1
            ? 3. obj.initial --> Static Method
            4. start&end declare
            5. formulation of functions
            6. why length cost der?
            7. why - 0.5
            """


### define our robot environment ###
class Environment():
    def __init__(self, pos, size, bound, voxel_size=np.array([0.1, 0.1]), eps=1, order=1):
        """
        Args:
            pos: np array with shape [N,2], with N being number of obstacles, indicating coordinate of obstacle'slower left corner
            size: np array with shape [N,2], with N being number of obstacles, indicating width and hight of obstacles
            bound: np array with shape [2,], upper boundary of the work space. Lower bound is implicitly (0,0)
            voxel_size: np array with shape [2,], voxel_size in x and y direction
            eps: scalar, parameter in cost function
            order: positive int, interpolation order
        """
        self.pos = pos.astype(int)
        self.size = size.astype(int)
        self.bound = bound.astype(int)
        self.voxel_size = voxel_size
        self.ob_num = pos.shape[0]
        self.eps = eps
        self.order = order

        self.obstacle = self.obstacle()
        self.dis = self.dis()
        self.dis_der = self.dis_der()
        self.dis_fun = self.dis_fun1()
        self.dis_der_fun = self.dis_der_fun1()
        self.cost_fun = self.cost_fun1()
        self.cost_der_fun = self.cost_der_fun1()

    def obstacle(self):
        """
        Geometric shape of the environment
        Returns:
            obstacle: a boolean numpy array with shape [bound[0],bound[1]], True indicates obstacle, False indicates free
        """
        pos = self.pos
        size = self.size
        bound = self.bound
        obstacle = np.zeros(bound, dtype=bool)
        for i in range(pos.shape[0]):
            low_left = pos[i]
            up_right = low_left + size[i]
            obstacle[tuple(map(slice, low_left, up_right))] = True
        return obstacle

    def dis(self):
        """
        Create nearest distance field, negative indicates inside obstacle
        Returns:
            dis: a float numpy array with shape [bound[0],bound[1]]
        """
        bound = self.bound
        voxel_size = self.voxel_size

        im = self.obstacle

        pad = np.ones(self.bound + 2, dtype=bool)
        pad[1:bound[0] + 1, 1:bound[1] + 1] = im

        dis = ndimage.distance_transform_edt(-pad.astype(int) + 1, sampling=voxel_size)
        dis1 = ndimage.distance_transform_edt(pad.astype(int), sampling=voxel_size)
        dis[pad] = - dis1[pad]  # Add interior information

        dis = dis[1:bound[0] + 1, 1:bound[1] + 1]
        return dis

    def dis_der(self):
        """
        Applying sobel filter to nearest distance field to get and x and y gradient field
        Returns:
            dis_der: a float numpy array with shape [2,bound[0],bound[1]], dis_der[0] x gradient and dis_der[1] y gradient
        """
        dis_der = np.zeros((2, self.bound[0], self.bound[1]), dtype=np.float64)
        for d in range(2):  # Treat image boundary like obstacle
            dis_der[d, ...] = ndimage.sobel(self.dis, axis=d, mode='constant', cval=0) / self.voxel_size[d]
        return dis_der

    def dis_fun1(self):
        """
        Interpolate the nearest distance to get distance function
        Returns:
            dis_fun: a function whose input is float numpy array with shape [N,2], N is number of inquiry points
                                      output is float numpy array with shape [N,], respecting cost of each inquiry points
        """
        factor = 1 / self.voxel_size
        im = self.dis

        def dis_fun(x):
            x = np.multiply(x, factor) - 0.5
            out = ndimage.map_coordinates(im, coordinates=x.T, order=self.order, mode='nearest')
            return out

        return dis_fun

    def dis_der_fun1(self):
        """
        Interpolate the x and y gradient field to get distance gradient function
        Returns:
            dis_der_fun: a function whose input is float numpy array with shape [N,2], N is number of inquiry points
                                          output is float numpy array with shape [N,2], respecting x and y gradient of each point
        """
        der = self.dis_der
        factor = 1 / self.voxel_size

        def dis_der_fun(x):
            x = np.multiply(x, factor) - 0.5
            gx = ndimage.map_coordinates(der[0, ...], coordinates=x.T, order=self.order, mode='nearest')
            gy = ndimage.map_coordinates(der[1, ...], coordinates=x.T, order=self.order, mode='nearest')
            return np.stack((gx, gy), axis=0).T

        return dis_der_fun

    def cost_fun1(self):
        """
        Assign cost to nearest distance field
        Returns:
            cost_fun: a function whose input is float numpy array with shape [N,2], N is number of inquiry points
                                       output is float numpy array with shape [N,], cost of each point
        """
        eps = self.eps

        def cost_fun(x):
            dis = self.dis_fun(x)
            cost = np.zeros(dis.shape, dtype=np.float64)
            cost[dis > eps] = 0
            cost[np.logical_and(dis > 0, dis <= eps)] = np.square(dis[np.logical_and(dis > 0, dis <= eps)] - eps) / (
                    2 * eps)
            cost[dis <= 0] = eps / 2 - dis[dis <= 0]
            return cost

        return cost_fun

    def cost_der_fun1(self):
        """
        Assign cost gradient
        Returns:
            cost_der_fun: a function whose input is float numpy array with shape [N,2], N is number of inquiry points
                                           output is float numpy array with shape [N,2], x and y cost gradient of each point
        """
        eps = self.eps

        def cost_der_fun(x):
            dis = self.dis_fun(x)
            dis_der = self.dis_der_fun(x)
            der = cost = np.zeros((len(dis), 2), dtype=np.float64)
            der[dis > eps] = 0
            der[np.logical_and(dis > 0, dis <= eps)] = np.multiply((dis[np.logical_and(dis > 0, dis <= eps)] - eps),
                                                                   dis_der[
                                                                       np.logical_and(dis > 0, dis <= eps)].T).T / eps
            der[dis <= 0] = - dis_der[dis < 0]
            return der

        return cost_der_fun


### setup reinforcement learning environment ###
class MPEnv(gym.Env):
    def __init__(self, objective):
        """
        Args:
            objective: Objective object that the reinforcement learning framework is based on
        """
        super(MPEnv, self).__init__()
        self.obj = objective  # objective function that RI is based on
        self.environment = self.obj.environment  # robot environment that RI is based on
        self.at_space_dim = self.obj.opt_num * 2  # action space dimension
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.at_space_dim,), dtype=np.float32)
        self.history_num = 1  # how many historical trajectory does the observation based on
        self.warp_dim = 2 * self.obj.opt_num  # dimension of observation that generated from one historical trajectory,
        # now we consider only gradient for easier training,
        # in future we can add objective value and coordinate
        self.ob_space_dim = self.history_num * self.warp_dim  # total dimension of observation space, start, end point coordinate not included,
        # may add that in future
        self.observation_space = spaces.Box(low=-40, high=40, shape=(self.ob_space_dim,), dtype=np.float32)
        self.pos = self.obj.initial()  # coordinate of the trajectory with shape [opt_num, 2], the state for RI
        self.observation = None

    def reset(self):
        """
        initialize the trajectory, return the observation initial state
        """
        self.pos = self.obj.initial()
        start = self.obj.start
        end = self.obj.end
        initial_der = self.obj.ob_der_fun(self.pos).flatten()
        initial_ob_val = np.array([self.obj.ob_fun(self.pos)])  # not used if history_num == 1
        initial_ob_val = np.array([self.obj.ob_fun(self.pos)])  # not used if history_num == 1
        history = np.zeros((self.history_num - 1) * (1 + 4 * self.obj.opt_num),
                           dtype=np.float32)  # not used if history_num == 1
        self.observation = np.concatenate((history, initial_der), axis=0).astype(np.float32)
        return self.observation

    def step(self, action):
        """
        Conduct the given action, go to the next state and return new observation, immediate reward, additional info and check if the goal is reached
        PS: done is actually useless for training, because reward is not depend on done, unlike other RI cases
        """
        self.pos = self.pos + action.reshape(self.obj.opt_num, 2)
        self.pos = np.clip(self.pos, 0, self.environment.bound * self.environment.voxel_size)
        obj_der = self.obj.ob_der_fun(self.pos).flatten()
        new_observation = obj_der
        self.observation = np.delete(self.observation, range(0, self.warp_dim))  # not used if history_num == 1
        self.observation = np.concatenate((self.observation, new_observation), axis=0).astype(
            np.float32)  # not used if history_num == 1
        # if np.isclose(new_observation, np.zeros_like(new_observation), rtol= 0.2).all():
        #     done = True
        # else:
        #     done = False
        done = bool(self.obj.collision(self.pos))
        reward = -self.obj.ob_fun(self.pos)
        info = {}
        return self.observation, reward, done, info

    def close(self):
        pass
