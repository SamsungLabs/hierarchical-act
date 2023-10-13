from abc import *
from typing import Dict, List

import gin
import numpy as np
import torch

import envs
import models

from ..base import BaseAgent
from languages import language_embedding as lang

@torch.jit.ignore
def to_numpy(x) -> float:
    return float(x.cpu().detach().numpy()[0])


@gin.configurable(denylist=['env', 'model'])
class HactVqAgent(BaseAgent):
    def __init__(self,
                env: envs.BaseEnv,
                model: models.BaseModel,
                max_timesteps: int=400,
                pred_is_pad_threshold: float=0.5,
                num_samples: int=10,
                temporal_agg_temp: float=0.01):
        super(HactVqAgent, self).__init__(env, model)
        self.env = env
        self.model = model
        self.chunk_size=self.model.chunk_size
        self.max_timesteps = max_timesteps
        self.state_dim = self.model.state_dim
        self.action_dim = self.model.action_dim
        self.pred_is_pad_threshold = pred_is_pad_threshold
        self.num_samples = num_samples
        self.temporal_agg_temp = temporal_agg_temp

        self._all_time_actions = torch.zeros([self.max_timesteps, 
                                                self.max_timesteps+self.chunk_size, 
                                                self.action_dim], 
                                                requires_grad=False).to("cuda")
        self._all_time_pad = torch.zeros([self.max_timesteps, 
                                                self.max_timesteps+self.chunk_size, 
                                                1], 
                                                requires_grad=False).to("cuda")

    def forward(self, 
                obs: Dict[str, torch.Tensor], 
                timestep:int,
                context:List[float]):
        qpos, images, context = self.get_model_input_from_obs(obs, context)
        if timestep == 0:
            self._all_time_actions = 0*self._all_time_actions
            self._all_time_pad = 0*self._all_time_pad
        
        model_outputs = self.model.forward(qpos, 
                                        images, 
                                        context,
                                        num_samples=10)
    
        actions = self.get_action_from_model_output(model_outputs,
                                                    timestep)
        return actions

   
    def get_exp_weights(self, actions_for_curr_step):
        k = self.temporal_agg_temp
        exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step))).to("cuda")
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights.unsqueeze(dim=1)
        return exp_weights
    

    def get_final_actions(self,
              model_outputs: Dict[str, torch.Tensor],
              timestep: int):

        pred_a = model_outputs['pred_a']
        pred_is_pad = model_outputs['pred_is_pad']
        pred_a[pred_is_pad[:,:,0]>self.pred_is_pad_threshold]=0
        
        with torch.no_grad():
            self._all_time_actions[[timestep], 
                timestep:timestep + self.chunk_size] = pred_a[:, :self.chunk_size, :]  # [1, action_chunk, state_dim] 
            
        actions_for_curr_step = self._all_time_actions[:, timestep]
        actions_populated = torch.all(actions_for_curr_step != 0, dim=1)  # [400] #boolean (True @ t)
        
        actions_for_curr_step = actions_for_curr_step[actions_populated]  # [t, 14]
        exp_weights = self.get_exp_weights(actions_for_curr_step)
        final_actions = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        return final_actions

    @abstractmethod
    def get_model_input_from_obs(self):
        raise NotImplementedError

    @abstractmethod
    def get_action_from_model_output(self):
        raise NotImplementedError