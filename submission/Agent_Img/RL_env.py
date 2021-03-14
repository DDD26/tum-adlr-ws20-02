import gym
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import numpy as np

class RLEnv(gym.Env):
    def __init__(self,MPEnv,sup_dim=30,sup_lr=0.2):
        super(RLEnv, self).__init__()
        self.MPEnv = MPEnv                                                             
        
        ### define action space 
        self.action_space = spaces.Box(low=-0.2,high=0.2,shape=(self.MPEnv.opt_num*2,),dtype=np.float32)

        ### define observation space                                                                                                                             
        self.observation_space = spaces.Box(low=0,high=3,shape=(64,64),dtype=np.float32)                                        
        
        self.pos = None
        self.reset()
        self.sup_dim = sup_dim
        self.sup_lr = sup_lr
        
        
    def gen_obs(self,pos):
        obs = self.MPEnv.obstacle.astype(np.uint8)
        pix = self.MPEnv.all_points(pos,5)
        pix = self.MPEnv.real2pix(pix)
        pix = np.unique(pix,axis=0)
        obs[pix.T[0],pix.T[1]] += 2
        return obs.astype(np.float32)
    
    def reset(self, random_start=False):
        """
        initialize the trajectory, return the observation of initial state
        """
        self.pos = self.MPEnv.initial(random_start)
        obs = self.gen_obs(self.pos)
        return obs
    
    def step(self, action):
        """
        Conduct the given action, go to the next state and return new observation, immdiate reward, additional info and check if goal is reached
        """
        ## update state
        self.pos = self.pos + action.reshape(self.MPEnv.opt_num,2)
        self.pos = np.clip(self.pos,0,self.MPEnv.bound*self.MPEnv.voxel_size)
        ## calculate observation
        obs = self.gen_obs(self.pos)
        ## calculate reward 
       
        """"
        start = self.real2pix(self.obj.start) 
        end = self.real2pix(self.obj.end) 
        num = 5
        pos1 = np.insert(self.pos,(0,self.pos.shape[0]),(start,end),axis=0)
        for i in range(pos1.shape[0]-1):
            pos2 = np.linspace(pos1[i+(num)*i],pos1[i+1+(num)*i],num+1,endpoint=False)
            pos2 = np.delete(pos2,0,0)
            pos1 = np.insert(pos1,i+1+(num)*i,pos2,axis=0)
        pos1 = np.clip(pos1,0,63)
        
        reward = 0
        for i in range(pos1.shape[0]):
            reward -= self.sdt[0][pos1[i][0]][pos1[i][1]]
        
        done = self.obj.collision(self.pix2real(self.pos))
        """
        """"
        reward = np.float(-np.sum(np.uint8(obs)==3))
        x = self.pos
        start = self.MPEnv.start
        end = self.MPEnv.end
        reward -= np.sum(np.diff(np.insert(x,(0,x.shape[0]),(self.MPEnv.pix2real(start),self.MPEnv.pix2real(end)),axis=0),axis=0)**2)
        """
        reward = -self.MPEnv.ob_fun(self.pos)
        done = self.MPEnv.collision(self.pos)  
        info = {}
        return obs, reward, done, info

    def close(self):
        pass
    
    def supervision(self):
        pos = self.MPEnv.initial()
        exp_obs = np.empty([self.sup_dim]+list(self.observation_space.shape))
        exp_act = np.empty([self.sup_dim]+list(self.action_space.shape))
        for i in range(self.sup_dim):
            action = -self.sup_lr*self.MPEnv.ob_der_fun(pos)
            exp_obs[i] = self.gen_obs(pos)
            exp_act[i] = action.flatten() 
            noise = np.random.normal(0,1,action.shape)
            pos = pos + (1+noise)*action
        return exp_obs, exp_act
        
        
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    
    def __init__(self, observation_space: gym.spaces.Box,features_dim: int = 64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.linear = nn.Sequential(nn.Linear(2304, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = torch.unsqueeze(observations,1)
        return self.linear(self.cnn(observations))

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
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ELU(),nn.Linear(last_layer_dim_pi,last_layer_dim_pi),nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ELU(),nn.Linear(last_layer_dim_vf,last_layer_dim_vf),nn.Tanh()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
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
        self.mlp_extractor = CustomNetwork(self.features_dim)