import numpy as np
from motion_env import MPEnv
from RL_env import RLEnv
from typing import Callable
from torch.utils.data.dataset import Dataset, random_split
from stable_baselines3.common.utils import set_random_seed
import gym


## helper function - generate train test environment ## 
def generate_env(ob_num, limit, opt_num, bound=np.array([64,64]),sup_dim = 30, sup_lr = 0.2):
    pos = np.empty([ob_num,2])
    pos[:,0] = np.random.randint(0,bound[0],ob_num)
    pos[:,1] = np.random.randint(0,bound[1],ob_num)
    size = np.random.randint(limit[0],limit[1],(ob_num,2))
    collision = True
    while collision:
        start = np.random.randint(1,bound[0]-1,2)
        end = np.random.randint(1,bound[1]-1,2)
        low = pos
        high = pos + size
        col1 = np.any(np.all(np.concatenate((start>low-1,start<high+1),axis=1),axis=1))
        col2 = np.any(np.all(np.concatenate((end>low-1,end<high+1),axis=1),axis=1))
        collision = col1 or col2
    MPenv = MPEnv(pos,size,bound,start,end,opt_num=opt_num)
    return RLEnv(MPenv,sup_dim=sup_dim,sup_lr=sup_lr)

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

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
        
    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)