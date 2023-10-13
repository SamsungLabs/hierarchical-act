from typing import Dict

import gin
import torch


import envs
import models

from .base import MultiActAgent
from .registry import register

@register('mujoco_bimanipulation')
@gin.configurable(denylist=['env', 'model'])
class MujocoBimanipulationMultiActAgent(MultiActAgent):
    def __init__(self,
                env: envs.BaseEnv,
                model: models.BaseModel,
                query_frequency: int=1,
                temporal_agg_temp: float=0.04,
                max_timesteps: int=400):
        super(MujocoBimanipulationMultiActAgent, self).__init__(
            env, 
            model,
            query_frequency=query_frequency,
            temporal_agg_temp=temporal_agg_temp,
            max_timesteps=max_timesteps)


    def get_model_input_from_obs(self,
                            obs: Dict[str, torch.Tensor],
                            description: str):
        qpos = torch.cat([
                            obs['qpos/left_arm_qpos'],
                            obs['qpos/left_gripper_positions'],
                            obs['qpos/right_arm_qpos'],
                            obs['qpos/right_gripper_positions']
                        ])
        qpos = qpos[None, :]
        qpos = qpos.to(torch.float32)

        images = []
        images.append(obs['images/top'])
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
        actions["left_arm_qpos"] = final_actions[0, 0:6]
        actions["left_gripper_positions"] = final_actions[0, 6:7]
        actions["right_arm_qpos"] = final_actions[0, 7:13]
        actions["right_gripper_positions"] = final_actions[0, 13:14]
        return actions