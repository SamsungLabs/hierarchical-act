from typing import Final

import numpy as np
import torch
from gym import spaces 

from .registry import register
from .base import BaseEnv

@register('srrc_dual_frankas')
class SrrcDualFrankasEnv(BaseEnv):
    def __init__(self):
        super(SrrcDualFrankasEnv, self).__init__()
        self.observation_space = self.build_observation_space()
        self.action_space = self.build_action_space()
        # self.command_space = self.build_command_space()
        
    def build_observation_space(self):
        return spaces.Dict(
            {
                "joint_position/left_arm": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32),
                "joint_position/right_arm": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32),
                "joint_position/left_hand": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "joint_position/right_hand": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                
                "images/left_head": spaces.Box(low=0, high=255, shape=(480,640,3), dtype=np.uint8),
                "images/right_head": spaces.Box(low=0, high=255, shape=(480,640,3), dtype=np.uint8),
                "images/left_hand": spaces.Box(low=0, high=255, shape=(480,640,3), dtype=np.uint8),
                "images/right_hand": spaces.Box(low=0, high=255, shape=(480,640,3), dtype=np.uint8),
            
                "eef_pose/left_eef": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                "eef_pose/right_eef": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )
    
    def build_action_space(self):
        return spaces.Dict(
            {
                "joint_command/left_arm": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32),
                "joint_command/right_arm": spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32),
                "joint_command/left_hand": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
                "joint_command/right_hand": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            }
        )