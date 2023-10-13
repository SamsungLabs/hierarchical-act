from abc import *
from typing import Dict, Final, Optional

import torch

import envs
import models
 

class BaseAgent(torch.nn.Module):
    def __init__(self,
                    env: envs.BaseEnv,
                    model: models.BaseModel):
        super(BaseAgent, self).__init__()
        self.env = env
        self.model = model

    @abstractmethod
    def forward(self, 
                obs: Dict[str, torch.Tensor], 
                timestep:int,
                description:str):
        raise NotImplementedError

    @abstractmethod
    def get_model_input_from_obs(self):
        raise NotImplementedError

    @abstractmethod
    def get_action_from_model_output(self):
        raise NotImplementedError
