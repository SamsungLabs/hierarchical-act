from typing import Final

import numpy as np
import torch
from gym import spaces

from .registry import register
from .base import BaseEnv

@register('maniskill2')
class Maniskill2Env(BaseEnv):
    def __init__(self):
        super(Maniskill2Env, self).__init__()
        self.observation_space = self.build_observation_space()
        self.action_space = self.build_action_space()
        
    def build_observation_space(self):
        return spaces.Dict(
            {
                "agent/qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
                "agent/qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
                "image/base_camera/rgb": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
                "image/hand_camera/rgb": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            }
        )

    def build_action_space(self):
        return spaces.Box(
            low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.], dtype=np.float32),
            high=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.], dtype=np.float32),
            shape=(8,))
    
