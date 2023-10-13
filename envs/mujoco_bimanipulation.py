from typing import Final

import numpy as np
import torch
from gym import spaces 

from .registry import register
from .base import BaseEnv

@register('mujoco_bimanipulation')
class MujocoBimanipulationEnv(BaseEnv):
    def __init__(self):
        super(MujocoBimanipulationEnv, self).__init__()
        self.observation_space = self.build_observation_space()
        self.action_space = self.build_action_space()
        
    def build_observation_space(self):
        return spaces.Dict(
            {
                "qpos/left_arm_qpos": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
                "qpos/left_gripper_positions": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "qpos/right_arm_qpos": spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32),
                "qpos/right_gripper_positions": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "qvel/left_arm_qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "qvel/left_gripper_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "qvel/right_arm_qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "qvel/right_gripper_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "images/top": spaces.Box(low=0, high=255, shape=(480,640,3), dtype=np.uint8),
            }
        )
    
    def build_action_space(self):
        return spaces.Dict(
            {
                "left_arm_qpos": spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(6,)),
                "left_gripper_positions": spaces.Box(low=0, high=1, shape=(1,)),
                "right_arm_qpos": spaces.Box(low=-np.pi/2, high=np.pi/2, shape=(6,)),
                "right_gripper_positions": spaces.Box(low=0, high=1, shape=(1,)),
            }
        )
    
        
    def obs_to_act_input(self):
        pass


    def at_ouput_to_action(self):
        pass