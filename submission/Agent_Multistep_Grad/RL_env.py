import gym
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import numpy as np

class RLEnv(gym.Env):
    def __init__(self,MPEnv,step,sup_dim=30,sup_lr=0.1):
        super(RLEnv, self).__init__()
        self.MPEnv = MPEnv                                                             
        self.sup_dim = sup_dim
        self.sup_lr = sup_lr
        self.step_num = step
        self.cur_step = 1
        
        ### define action space 
        self.action_space = spaces.Box(low=-0.2,high=0.2,shape=(self.MPEnv.opt_num*2,),dtype=np.float32)
        
        ### define observation space                                                                                                                     
        self.ob_space_dim = self.step_num*(self.MPEnv.opt_num*2+1)
        self.observation_space = spaces.Box(low=-5,high=5,shape=(self.ob_space_dim,),dtype=np.float32)                                        
        
        self.pos = None
        self.state = np.zeros((self.step_num,self.MPEnv.opt_num,2),dtype=np.float32)
        self.reset()
        
    def gen_obs(self,state,cur_step):
        obs = np.zeros((self.observation_space.shape),dtype=np.float32)
        for i in range(np.max([0,self.step_num-cur_step]),self.step_num):
            obs[i*(2*self.MPEnv.opt_num+1)] = self.MPEnv.ob_fun(state[i]) -  self.MPEnv.ob_fun(state[-1])
            obs[i*(2*self.MPEnv.opt_num+1)+1:(i+1)*(2*self.MPEnv.opt_num+1)] = self.MPEnv.ob_der_fun(state[-1-i]).flatten() 
        return obs
    
    def reset(self, random_start=False):
        """
        initialize the trajectory, return the observation of initial state
        """
        self.pos = self.MPEnv.initial(random_start)
        self.state[-1] = self.pos
        self.cur_step = 1
        obs = self.gen_obs(self.state,self.cur_step)    
        return obs.astype(np.float32)
    
    def step(self, action):
        """
        Conduct the given action, go to the next state and return new observation, immdiate reward, additional info and check if goal is reached
        """
        ## update state
        self.pos = self.pos + action.reshape(self.MPEnv.opt_num,2)
        self.pos = np.clip(self.pos,0,self.MPEnv.bound*self.MPEnv.voxel_size)
        self.state = np.delete(self.state,0,axis=0)
        self.state = np.append(self.state,np.expand_dims(self.pos,axis=0),axis=0)
        self.cur_step += 1
        ## calculate observation
        obs = self.gen_obs(self.state,self.cur_step)
        ## calculate reward
        done = bool(self.MPEnv.collision(self.pos))
        reward = -self.MPEnv.ob_fun(self.pos)
        info = {}
        return obs,reward,done,info

    def close(self):
        pass
    
    def supervision(self):
        pos = self.MPEnv.initial()
        state = np.zeros((self.step_num,self.MPEnv.opt_num,2),dtype=np.float32)
        state[-1] = pos
        cur_step = 1
        exp_obs = np.empty([self.sup_dim]+list(self.observation_space.shape))
        exp_act = np.empty([self.sup_dim]+list(self.action_space.shape))
        for i in range(self.sup_dim):
            exp_obs[i] = self.gen_obs(state,cur_step)
            exp_act[i] = -self.sup_lr*self.MPEnv.ob_der_fun(pos).flatten()
            noise = np.random.normal(0,1,(self.MPEnv.opt_num,2))
            pos = pos + (1+noise)*exp_act[i].reshape(self.MPEnv.opt_num,2)
            state = np.delete(state,0,axis=0)
            state = np.append(state,np.expand_dims(pos,axis=0),axis=0)
            cur_step += 1
        return exp_obs, exp_act
    

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int ,
        last_layer_dim_pi: int = 10,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.linear1 = nn.Linear(feature_dim-10,64)
        self.linear2 = nn.Linear(64,10)
        self.linear3 = nn.Linear(feature_dim-10,64)
        self.linear4 = nn.Linear(64,64)
        self.act = nn.Softplus()

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad = features[:,-10:]
        pi = features[:,:-10]
        pi = self.linear1(pi)
        pi = self.act(pi)
        pi = self.linear2(pi)
        pi = 0.2*torch.tanh(pi)
        pi = pi - 0.1*grad
        
        vf = features[:,:-10]
        vf = self.linear3(vf)
        vf = self.act(vf)
        vf = self.linear4(vf)
        vf = self.act(vf)
        return pi, vf


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
        self.mlp_extractor = CustomNetwork(self.features_dim)