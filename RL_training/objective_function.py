import numpy as np

import scipy.ndimage as ndimage
import torch
import gym
from gym import spaces
from stable_baselines import PPO2
# from stable_baselines3 import PPO, A2C, SAC # DQN coming soon
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

class Objective():
    def __init__(self, start, end, opt_num, sp_num, co_num, environment, w):
        """
        Args:
            start: np array with shape [2,], start point coordinate of the robot
            end: np: array with shape [2,], end point coordinate of the robot
            opt_num: number of optimization points
            sp_num: number of subsampling points on line segements between two optimization points for calculating objective
            co_num: number of subsampling points on line segements between two optimization points for collision check
            environment: environment the objective function based on
            w: weight term for length objective
        """
        self.start = start
        self.end = end
        self.opt_num = opt_num
        self.sp_num = sp_num
        self.co_num = co_num
        self.environment = environment
        self.w = w  # length weight

        self.ob_fun = self.ob_fun1()
        self.ob_der_fun = self.ob_der_fun1()

    def ob_fun1(self):
        """
        Given a trajectory, calculate its obstacle cost and length cost and objective
        Returns:
            ob_fun: a function whose input is float numpy array with shape [opt_num, 2]
                                     output is float numpy scalar, the objective value
        """
        env = self.environment
        w = self.w
        start = self.start
        end = self.end

        def ob_fun(x):
            x1 = self.all_points(x, self.sp_num)
            x1 = np.delete(x1, 0, 0)
            x1 = np.delete(x1, x1.shape[0] - 1, 0)
            return np.mean(env.cost_fun(x1)) + w * np.sum(
                np.diff(np.insert(x, (0, x.shape[0]), (start, end), axis=0), axis=0) ** 2)

        return ob_fun

    def ob_der_fun1(self):
        """
        Derivative of objective function
        Returns:
            ob_der_fun: a function whose input is a float numpy array with shape [opt_num, 2]
                                         output is a float numpy array with shape [opt_num,2], the derivative
        """
        env = self.environment
        w = self.w
        opt_num = self.opt_num
        sp_num = self.sp_num

        def ob_der_fun(x):
            ### gradient of obstacle cost ###
            x1 = self.all_points(x, self.sp_num)
            x1 = np.delete(x1, 0, 0)
            x1 = np.delete(x1, x1.shape[0] - 1, 0)
            x1 = self.environment.cost_der_fun(x1)
            x1 = torch.Tensor(x1).reshape(1, 1, x1.shape[0], x1.shape[1])
            kernel1 = np.append(np.arange(1, sp_num + 2, 1), np.arange(sp_num, 0, -1)) / (sp_num + 1)
            kernel1 = torch.Tensor(kernel1).reshape(1, 1, kernel1.shape[0], 1)
            re1 = torch.nn.functional.conv2d(x1, kernel1, stride=(sp_num + 1, 1))
            re1 = re1 / (opt_num + (opt_num + 1) * sp_num)
            re1 = torch.squeeze(torch.squeeze(re1, 0), 0).numpy()
            ### gradient of length cost ###
            x2 = np.insert(x, (0, x.shape[0]), (self.start, self.end), axis=0)
            x2 = torch.Tensor(x2).reshape(1, 1, x2.shape[0], x2.shape[1])
            kernel2 = torch.Tensor([-1, 2, -1]).reshape(1, 1, 3, 1)
            re2 = 2 * w * torch.nn.functional.conv2d(x2, kernel2, stride=1)
            re2 = torch.squeeze(torch.squeeze(re2, 0), 0).numpy()
            return re1 + re2

        return ob_der_fun

    def all_points(self, x, num):
        """
        Combine all start, end, optimization and subsampling points (both for calculating objective and collision check)
        Args:
            x: float numpy array with shape [opt_num,2], optimization points
            num: number of subsampling points
        Returns:
            x1: float numpy array with shape [opt_num+2+num*(opt_num+1), 2]
        """
        start = self.start
        end = self.end
        x1 = np.insert(x, (0, x.shape[0]), (start, end), axis=0)
        for i in range(x1.shape[0] - 1):
            x2 = np.linspace(x1[i + (num) * i], x1[i + 1 + (num) * i], num + 1, endpoint=False)
            x2 = np.delete(x2, 0, 0)
            x1 = np.insert(x1, i + 1 + (num) * i, x2, axis=0)
        return x1

    def initial(self):
        """
        Initialize the trajectory by connecting start, end point and uniform sampling along this line segment
        Returns:
            x0: float numpy array with shape [opt_num, 2], initial optimization points
        """
        x0 = np.linspace(self.start, self.end, self.opt_num + 1, endpoint=False)
        x0 = np.delete(x0, 0, 0)
        return x0

    def collision(self, x):
        """
        Check if any of optimization and subsampling points collides with any of the obstacles. Moreover check if all points are in the boundary.
        If both conditions are satisfied, returns True, otherwise False.
        """
        low = self.environment.pos
        high = self.environment.pos + self.environment.size
        x1 = self.all_points(x, self.co_num)
        factor = 1 / self.environment.voxel_size
        x1 = np.multiply(x1, factor) - 0.5
        out = np.empty((x1.shape[0],), dtype=bool)
        for i in range(x1.shape[0]):
            k = np.concatenate((x1[i] > low, x1[i] < high), axis=1)
            k = np.all(k, axis=1)
            out[i] = np.any(k)
            out1 = np.any(out)
            out2 = np.all([x1 > 0, x1 < self.environment.bound])
        return not out1 and out2
