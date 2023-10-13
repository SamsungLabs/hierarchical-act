from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np

@dataclass
class EpisodicDataClass:
    description: str
    sentence_embedding: np.array
    episode_len: int
    qpos: Dict[str, np.array]
    qvel: Dict[str, np.array]
    images: Dict[str, np.array]
    actions: Dict[str, np.array]


@dataclass
class DataStatistics:
    is_sim: Optional[str] = None
    action_mean: Optional[np.array] = None
    action_std: Optional[np.array] = None
    qpos_mean: Optional[np.array] = None
    qpos_std: Optional[np.array] = None
    qvel_mean: Optional[np.array] = None
    qvel_std: Optional[np.array] = None