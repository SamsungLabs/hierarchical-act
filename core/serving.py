import os
from typing import Dict

import gin
import torch
import numpy as np

import agents
import envs
from core import Task
from common import nested_tensor_utils
from languages import language_embedding as lang

class Serving():
    def __init__(self,
                 task:Task,
                 gpu:int=0):
        self.task = task
        self.gpu = gpu
        self.env = self.task.get_env()
        self._device = None

    def serve(self):
        if not torch.cuda.is_available():
            raise Exception("Only cuda device is supported for serving")
        else:
            torch.cuda.set_device(self.gpu)
            self._device = torch.device("cuda")

        exp_dir = self.task.exp_dir
        model_path = os.path.join(exp_dir, 'model_scripted.pt')
        agent_path = os.path.join(exp_dir, 'agent.pt')
        
        self.task.restore_model()
        model = self.task.get_model(self.gpu)
        env = self.env
        
        agent = agents.make(self.task.model_id, self.task.env_id)(env, model)

        self.check_inputoutput(agent)
        agent_scripted = torch.jit.script(agent)
        agent_scripted.save(agent_path)
        
    def check_inputoutput(self, agent):
        obs = self.env.observation_space.sample()
        sampled_actions = self.env.action_space.sample()
        description = 'straw insertion'

        torch_obs = nested_tensor_utils.to_torch(obs, )
        torch_obs = nested_tensor_utils.to_device(torch_obs, device=self._device)
        context = lang.binary_encoder(description)

        torch_actions = agent.forward(torch_obs, timestep=0, context=context)

        actions = nested_tensor_utils.to_numpy(torch_actions)
        if isinstance(sampled_actions, Dict):
            for key, value in sampled_actions.items():
                try:
                    #assert actions[key].dtype == value.dtype
                    assert actions[key].shape == value.shape
                except:
                    raise Exception(f"Mismatch {key} between env.action ({value.shape})"\
                                    f"and agent.action ({actions[key].shape})")
        else:
            try:
                #assert actions[key].dtype == value.dtype
                assert actions.shape == value.shape
            except:
                raise Exception(f"Mismatch {key} between env.action ({value.shape})"\
                                    f"and agent.action ({actions.shape})")