from abc import *
from collections import namedtuple
from typing import Dict, Tuple

import torch
from gym.spaces import Dict as Gym_Dict

#class BaseEnv(torch.jit.ScriptModule):
class BaseEnv():
    def __init__(self):
        super(BaseEnv, self).__init__()
        self.observation_space = self.build_observation_space()
        self.action_space = self.build_action_space()


    @torch.jit.unused
    def create_observations_from_dict(self, obs_dict:Dict):
        observations = {}
        for key, value in self.observation_space.items():
            if not key in obs_dict.keys():
                raise Exception(f"Key {key} not exists in observation_space")
            if not obs_dict[key].dtype == value.dtype:
                raise Exception(f"Observation dtype mismatch between space ({value.dtype}) and input_dict ({obs_dict[key].dtype})")
            if not obs_dict[key].shape[-len(value.shape):] == value.shape:
                raise Exception(f"Observation shape mismatch between space ({value.shape}) and input_dict ({obs_dict[key].shape})")
            observations[key] = obs_dict[key]
        return observations

    @torch.jit.unused
    def create_actions_from_dict(self, act_dict:Dict):
        actions = {}
        for key, value in self.action_space.items():
            if not key in act_dict.keys():
                raise Exception(f"Key {key} not exists in observation_space")
            if not act_dict[key].dtype == value.dtype:
                raise Exception(f"Action dtype mismatch between space ({value.dtype}) and input_dict ({act_dict[key].dtype})")
            if not act_dict[key].shape[-len(value.shape):] == value.shape:
                raise Exception(f"Action shape mismatch between space ({value.shape}) and input_dict ({act_dict[key].shape})")
            actions[key] = act_dict[key]
        return actions

    @abstractmethod
    def build_observation_space(self)->Gym_Dict:
        raise NotImplementedError

    @abstractmethod
    def build_action_space(self)->Gym_Dict:
        raise NotImplementedError