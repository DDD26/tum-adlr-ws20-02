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
        self.sup_dim = sup_dim
        self.sup_lr = sup_lr
        
        ### define action space 
        self.action_space = spaces.Box(low=-0.2,high=0.2,shape=(self.MPEnv.opt_num*2,),dtype=np.float32)

        ### define observation space                                                                                                                     
        self.observation_space = spaces.Box(low=0,high=3,shape=(2,64,64),dtype=np.float32)                                        
        
        self.pos = None
        self.reset()
        self.min_ob = None
        
    def gen_obs(self,pos):
        obs = np.zeros((self.observation_space.shape),dtype=np.float32)
        obs[0,0,0:self.MPEnv.opt_num*2] = -self.sup_lr*self.MPEnv.ob_der_fun(pos).flatten()
        obs[1] = self.MPEnv.obstacle.astype(np.uint8)
        pix = self.MPEnv.all_points(pos,5)
        pix = self.MPEnv.real2pix(pix)
        pix = np.unique(pix,axis=0)
        obs[1,pix.T[0],pix.T[1]] += 2
        return obs.astype(np.float32)
    
    def reset(self, random_start=False):
        """
        initialize the trajectory, return the observation of initial state
        """
        self.pos = self.MPEnv.initial(random_start)
        obs = self.gen_obs(self.pos)
        self.min_ob = self.MPEnv.ob_fun(self.pos)
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
        
        done = self.MPEnv.collision(self.pos)
        if done:
            reward = -0.1*self.MPEnv.ob_fun(self.pos)
        else: reward = -self.MPEnv.ob_fun(self.pos)
        
        '''
        if done:
            reward = 1
        else :
            if ob > self.min_ob:
                reward = 0.1
            else: 
                reward = -0.1 
                self.min_ob = ob
        '''
        info = {}
        return obs, reward, done, info

    def close(self):
        pass
    
    def supervision(self):
        pos = self.MPEnv.initial()
        exp_obs = np.empty([self.sup_dim]+list(self.observation_space.shape))
        exp_act = np.empty([self.sup_dim]+list(self.action_space.shape))
        for i in range(self.sup_dim):
            exp_obs[i] = self.gen_obs(pos)
            exp_act[i] = exp_obs[i,0,0,0:self.MPEnv.opt_num*2] 
            noise = np.random.normal(0,1,(self.MPEnv.opt_num,2))
            pos = pos + (1+noise)*exp_act[i].reshape(self.MPEnv.opt_num,2)
        return exp_obs, exp_act
        
    
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 20):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0) 
        self.linear = nn.Linear(2304,features_dim-20)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        grad = observations[:,0,0,0:20] 
        out = grad.view(observations.shape[0],-1)
        x = torch.unsqueeze(observations[:,1,:,:,],axis=1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(observations.shape[0],-1)
        x = self.linear(x)
        x = nn.functional.relu(x)
        out = torch.cat((grad,x),axis=1)
        return out

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
        last_layer_dim_pi: int = 20,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.linear1 = nn.Linear(feature_dim-20,64)
        self.linear2 = nn.Linear(64,20)
        self.linear3 = nn.Linear(feature_dim-20,64)
        self.linear4 = nn.Linear(64,64)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad = features[:,0:20]
        pi = features[:,20:]
        pi = self.linear1(pi)
        pi = torch.nn.functional.elu(pi)
        pi = self.linear2(pi)
        pi = 0.02*torch.nn.functional.tanh(pi)
        pi = pi + grad
        
        vf = features[:,20:]
        vf = self.linear3(vf)
        vf = torch.nn.functional.elu(vf)
        vf = self.linear4(vf)
        vf = torch.nn.functional.tanh(vf)
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