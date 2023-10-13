from typing import Dict

import gin
import torch


import envs
import models

from .base import MultiActAgent
from .registry import register

@register('srrc_dual_frankas')
@gin.configurable(denylist=['env', 'model'])
class SrrcDualFrankasMultiActAgent(MultiActAgent):
    def __init__(self,
                 env: envs.BaseEnv,
                 model: models.BaseModel,
                 query_frequency: int = 1,
                 temporal_agg_temp: float = 0.04,
                 max_timesteps: int = 400):
        super(SrrcDualFrankasMultiActAgent, self).__init__(
            env,
            model,
            query_frequency=query_frequency,
            temporal_agg_temp=temporal_agg_temp,
            max_timesteps=max_timesteps)

    def get_model_input_from_obs(self,
                                 obs: Dict[str, torch.Tensor],
                                 description: str):
        left_arm = obs["joint_position/left_arm"]
        left_hand = obs["joint_position/left_hand"]
        right_arm = obs["joint_position/right_arm"]
        right_hand = obs["joint_position/right_hand"]
        qpos = torch.concat([left_arm, left_hand,
                             right_arm, right_hand], dim=0)
        qpos = qpos[None, :]
        qpos = qpos.to(torch.float32)

        images = []
        images.append(obs['images/left_head'])
        images.append(obs['images/right_head'])
        images.append(obs['images/left_hand'])
        images.append(obs['images/right_hand'])
        images = torch.stack(images, dim=0)
        images = torch.permute(images, [0, 3, 1, 2])
        images = images[None]
        images = images.to(torch.float32)
        images = images / 255.

        device = qpos.device
        # contexts = self.get_embedding(description)
        # contexts = contexts[None]
        # contexts = torch.from_numpy(contexts)
        contexts = torch.zeros(512).to(device)
        contexts = contexts[None]
        return qpos, images, contexts

    def get_action_from_model_output(self,
                                     model_outputs: Dict[str, torch.Tensor],
                                     timestep:int,
                                     temporal_agg:bool=False):
        
        final_actions = self.get_final_actions(model_outputs, 
                                               timestep,
                                               temporal_agg=temporal_agg)
        actions = dict()
        actions["joint_command/left_arm"] = final_actions[0, 0:7]
        actions["joint_command/left_hand"] = final_actions[0, 7:8]
        actions["joint_command/right_arm"] = final_actions[0, 8:15]
        actions["joint_command/right_hand"] = final_actions[0, 15:16]
        return actions