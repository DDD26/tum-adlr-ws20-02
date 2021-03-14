### import ###
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split
from typing import Callable

import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, SAC # DQN coming soon
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


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
            der[dis <= 0] = - dis_der[dis <= 0]
            return der

        return cost_der_fun


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
        self.sup_dim = 100
        self.sup_lr = 0.1

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
        start = self.start
        end = self.end

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
            x2 = np.insert(x, (0, x.shape[0]), (start, end), axis=0)
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

    def initial(self, random_start=False):
        """
        Initialize the trajectory by connecting start, end point and uniform sampling along this line segment
        Returns:
            x0: float numpy array with shape [opt_num, 2] initial optimization points
        """
        if random_start:
            x0 = np.zeros((self.opt_num, 2))
            low_x, high_x = (1, self.environment.bound[0] - 1) * self.environment.voxel_size
            low_y, high_y = (1, self.environment.bound[1] - 1) * self.environment.voxel_size
            range_xy = np.array([high_x - low_x, high_y - low_y])
            for i in range(self.opt_num):
                x0[i, :] = np.random.random((1, 2)) * range_xy + np.array([low_x, low_y])
            x0.sort(axis=0)
        else:
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
        x1 = np.multiply(x1, 1 / self.environment.voxel_size) - 0.5
        out = np.empty((x1.shape[0],), dtype=bool)
        for i in range(x1.shape[0]):
            k = np.concatenate((x1[i] > low, x1[i] < high), axis=1)
            k = np.all(k, axis=1)
            out[i] = np.any(k)
            out1 = np.any(out)
        out2 = np.all([x1 > 0, x1 < self.environment.bound])
        return not out1 and out2


### setup reinforcement learning environment ###
class MPEnv(gym.Env):
    def __init__(self, objective, sup_dim, sup_lr):
        """
        Args:
            objective: Objective object that the reinforcement learning framework is based on
        """
        super(MPEnv, self).__init__()
        self.obj = objective  # objective function that RI is based on
        self.environment = self.obj.environment  # robot environment that RI is based on

        ### define action space
        self.at_space_dim = self.obj.opt_num * 2  # action space dimension
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.at_space_dim,), dtype=np.float32)

        ### define observation space
        self.ob_space_dim = 3 * self.obj.opt_num + 1
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.ob_space_dim,), dtype=np.float32)
        
        ### define state 
        self.pos = None
        
        ### define supervision data
        self.sup_dim = sup_dim
        self.sup_lr = sup_lr
        
        

    def reset(self, random_start=False):
        """
        initialize the trajectory, return the observation initial state
        """
        self.pos = self.obj.initial(random_start)
        current_der = self.obj.ob_der_fun(self.pos).flatten()
        current_cost = self.environment.cost_fun(self.pos).flatten()
        current_cost_value = np.array([np.mean(current_cost)])
        observation = np.concatenate((current_der, current_cost, current_cost_value), axis=0).astype(np.float32)
        return observation

    def step(self, action):
        """
        Conduct the given action, go to the next state and return new observation, immdiate reward, additional info and check if goal is reached
        PS: done is actually useless for training, because reward is not depend on done, unlike other RI cases
        """
        self.pos = self.pos + action.reshape(self.obj.opt_num, 2)
        self.pos = np.clip(self.pos, 0, self.environment.bound * self.environment.voxel_size)
        
        current_der = self.obj.ob_der_fun(self.pos).flatten()
        current_cost = self.environment.cost_fun(self.pos).flatten()
        current_cost_value = np.array([np.mean(current_cost)])
        
        obs = np.concatenate((current_der, current_cost, current_cost_value), axis=0).astype(np.float32)

        done = bool(self.obj.collision(self.pos))
        if done:
            # reward =1
            reward = -self.obj.ob_fun(self.pos) * 0.1
        else:
            reward = -self.obj.ob_fun(self.pos)
            # reward = 0
        info = {}
        return obs, reward, done, info

    def close(self):
        pass

    def supervision(self):
        np.random.seed(0)
        positions = np.random.rand(self.sup_dim, self.obj.opt_num, 2)
        low = np.array([0.5, 0.5]) * self.environment.voxel_size
        high = (self.environment.bound + 0.5) * self.environment.voxel_size
        positions = (high - low) * positions + low
        exp_obs = np.empty([self.sup_dim, self.ob_space_dim])
        exp_act = np.empty([self.sup_dim, 2 * self.obj.opt_num])
        for i in range(positions.shape[0]):
            current_der = self.obj.ob_der_fun(positions[i]).flatten()
            current_cost = self.environment.cost_fun(positions[i]).flatten()
            current_cost_value = np.array([np.mean(current_cost)])
            
            exp_obs[i] = np.concatenate((current_der, current_cost, current_cost_value), axis=0).astype(np.float32)
            exp_act[i] = -self.sup_lr * self.obj.ob_der_fun(positions[i]).flatten()
        return exp_obs, exp_act


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)


### helper function - pretrain agent ###
def pretrain_agent(
    student,
    exp_train,
    exp_test,
    batch_size=64,
    epochs=10,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=False,
    seed=1,
    test_batch_size=64,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    criterion = nn.MSELoss()
    # Extract initial policy
    model = student.policy.to(device)
    def train(model, device, train_loader, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            if isinstance(student, (A2C, PPO)):
                action, _, _ = model(data)
            else:
                # SAC/TD3:
                action = model(data)
            action_prediction = action

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch,batch_idx * len(data),
                       len(train_loader.dataset),100.0 * batch_idx / len(train_loader),loss.item()))
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    # SAC/TD3:
                    action = model(data)
                action_prediction = action
                test_loss = criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        print(f"Test set: Average loss: {test_loss:.4f}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = torch.utils.data.DataLoader(
        dataset=exp_train, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=exp_test, batch_size=test_batch_size, shuffle=True, **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model


## helper function - generate train test environment ##
def generate_env(ob_num, limit, opt_num, sup_dim, bound=np.array([64, 64]), sp_num=5, co_num=10, w=0.1, sup_lr=0.1):
    pos = np.empty([ob_num, 2])
    pos[:, 0] = np.random.randint(0, bound[0], ob_num)
    pos[:, 1] = np.random.randint(0, bound[1], ob_num)
    size = np.random.randint(limit[0], limit[1], (ob_num, 2))
    collision = True
    while collision:
        start = np.random.randint(1, bound[0] - 1, 2)
        end = np.random.randint(1, bound[1] - 1, 2)
        low = pos
        high = pos + size
        col1 = np.any(np.all(np.concatenate((start > low - 1, start < high + 1), axis=1), axis=1))
        col2 = np.any(np.all(np.concatenate((end > low - 1, end < high + 1), axis=1), axis=1))
        collision = col1 or col2
    environment = Environment(pos, size, bound)
    start = environment.voxel_size * (start + 0.5)
    end = environment.voxel_size * (end + 0.5)
    obj = Objective(start, end, opt_num, sp_num, co_num, environment, w)
    return MPEnv(obj, sup_dim, sup_lr)


def make_env(env, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        policy_fc1 = nn.Linear(31,128)
        nn.init.zeros_(policy_fc1.bias)
        policy_a1 = nn.Tanh()
        policy_fc2 = nn.Linear(128,128)
        nn.init.eye_(policy_fc2.weight)
        nn.init.zeros_(policy_fc2.bias)
        policy_a2 = nn.Tanh()
        policy_layers = [policy_fc1, policy_a1, policy_fc2, policy_a2]
        self.policy_net = nn.Sequential(*policy_layers)

        value_fc1 = nn.Linear(31, 128)
        nn.init.xavier_uniform_(value_fc1.weight)
        nn.init.zeros_(value_fc1.bias)
        value_a1 = nn.Tanh()
        value_fc2 = nn.Linear(128,128)
        nn.init.xavier_uniform_(value_fc2.weight)
        nn.init.zeros_(value_fc2.bias)
        value_a2 = nn.Tanh()
        value_layers = [value_fc1, value_a1, value_fc2, value_a2]
        self.value_net = nn.Sequential(*value_layers)

        self.latent_dim_pi = 128
        self.latent_dim_vf = 128

    def forward(self, features):
        return self.policy_net(features), self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork()
        
        
    ### runtime evaluation ###
def runtime(model, env_list):
        n_steps = 200
        lr = 0.1
        result_hard_GD = 0
        result_hard_RL = 0
        GD_iteration_list, RL_iteration_list = [], []
        for env_test in env_list:
            obs = env_test.reset()
            x0 = env_test.pos
            for step in range(n_steps):
                x0 = x0 - lr * env_test.obj.ob_der_fun(x0)
                if env_test.obj.collision(x0):
                    result_hard_GD += 1
                    GD_iteration_list.append(step)
                    break

            for step in range(n_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env_test.step(action)
                if done:
                    result_hard_RL += 1
                    RL_iteration_list.append(step)
                    break
            env_test.close()
        
        print('===========================================================\n')
        print("GD success rate:", len(GD_iteration_list)/len(env_list), "GD average iterations=", sum(GD_iteration_list)/len(GD_iteration_list))
        print("RL success rate:", len(RL_iteration_list)/len(env_list), "RL average iterations=", sum(RL_iteration_list)/len(RL_iteration_list))
        print('===========================================================\n')