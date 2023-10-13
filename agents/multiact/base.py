from abc import *
from typing import Dict

import gin
import torch

import envs
import models

from ..base import BaseAgent
from languages import language_embedding as lang

@gin.configurable(denylist=['env', 'model'])
class MultiActAgent(BaseAgent):
    def __init__(self,
                env: envs.BaseEnv,
                model: models.BaseModel,
                query_frequency: int=1,
                temporal_agg_temp: float=0.04,
                max_timesteps: int=400):
        super(MultiActAgent, self).__init__(env, model)
        self.env = env
        self.model = model
        self.query_frequency = query_frequency
        self.temporal_agg_temp = temporal_agg_temp
        self.max_timesteps = max_timesteps
        self.chunk_size = self.model.chunk_size
        self.state_dim = self.model.state_dim
        self.action_dim = self.model.action_dim
        self._all_time_actions = torch.zeros([self.max_timesteps, 
                                                self.max_timesteps+self.chunk_size, 
                                                self.action_dim], 
                                                requires_grad=False).to("cuda")
    
    def forward(self, 
                obs: Dict[str, torch.Tensor], 
                timestep:int,
                description:str,
                temporal_agg:bool=False):
        qpos, images, contexts = self.get_model_input_from_obs(obs, description)
        model_outputs = self.model.forward(qpos, images, contexts)
        actions = self.get_action_from_model_output(model_outputs, 
                                                    timestep, 
                                                    temporal_agg=temporal_agg)
        return actions
    

    def get_exp_weights(self, actions_for_curr_step):
        k = self.temporal_agg_temp
        exp_weights = torch.exp(-k * torch.arange(len(actions_for_curr_step))).to("cuda")
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = exp_weights.unsqueeze(dim=1)
        return exp_weights
    

    def get_final_actions(self, 
              model_outputs: Dict[str, torch.Tensor],
              timestep: int,
              temporal_agg: bool=False):
        
        a_pred = model_outputs['a_hat']
        if timestep % self.query_frequency == 0:
            with torch.no_grad():
                self._all_time_actions[[timestep],
                    timestep:timestep + self.chunk_size] = a_pred  # [1, action_chunk, state_dim]

        if temporal_agg:
            actions_for_curr_step = self._all_time_actions[:, timestep]
            actions_populated = torch.all(actions_for_curr_step != 0, dim=1)  # [400] #boolean (True @ t)
            actions_for_curr_step = actions_for_curr_step[actions_populated]  # [t, 14]

            exp_weights = self.get_exp_weights(actions_for_curr_step)
            final_actions = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        else:
            last_inferenced_timestep = int(self.query_frequency * (timestep // self.query_frequency))
            final_actions = self._all_time_actions[[last_inferenced_timestep], timestep, :]
        return final_actions

    @abstractmethod
    def get_model_input_from_obs(self):
        raise NotImplementedError

    @abstractmethod
    def get_action_from_model_output(self):
        raise NotImplementedError


