from typing import Dict, List

import gin
import torch


import envs
import models

from .base import HactVqAgent
from .registry import register

@register('mujoco_bimanipulation')
@gin.configurable(denylist=['env', 'model'])
class MujocoBimanipulationHactVqAgent(HactVqAgent):
    def __init__(self,
                env: envs.BaseEnv,
                model: models.BaseModel,
                max_timesteps: int=400,
                pred_is_pad_threshold: float=0.5,
                num_samples: int=10,
                temporal_agg_temp: float=0.01):
        super(MujocoBimanipulationHactVqAgent, self).__init__(
            env, 
            model,
            max_timesteps=max_timesteps,
            pred_is_pad_threshold=pred_is_pad_threshold,
            num_samples=num_samples,
            temporal_agg_temp=temporal_agg_temp)


    def get_model_input_from_obs(self,
                            obs: Dict[str, torch.Tensor],
                            context: List[float]):
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
        #contexts = self.get_sentence_embedding(description)
        #contexts = contexts[None]
        context = torch.FloatTensor(context).to(device)
        context = context[None]
        context = context / 255.
        #contexts = torch.zeros(512).to(device)
        return qpos, images, context

    def get_action_from_model_output(self,
                                     model_outputs: Dict[str, torch.Tensor],
                                     timestep:int):
        
        final_actions = self.get_final_actions(model_outputs, 
                                               timestep)
        actions = dict()
        actions["left_arm_qpos"] = final_actions[0, 0:6]
        actions["left_gripper_positions"] = final_actions[0, 6:7]
        actions["right_arm_qpos"] = final_actions[0, 7:13]
        actions["right_gripper_positions"] = final_actions[0, 13:14]
        return actions