import glob
import os
import random
from abc import *
from typing import List, Dict, Union

import torch

class BaseInput(torch.utils.data.IterableDataset):
    def __init__(self,
                 data_root:str,
                 data_rel_dir:str,
                 split:str='train',
                 task_list: Union[str, List[str]]='all',
                 num_train_demo: Union[str, int]='all',
                 num_val_demo: Union[str, int]='all',
                 shuffle=True):
        self.data_dir = os.path.join(data_root, data_rel_dir)
        self.split = split
        self.task_list = task_list
        self.num_train_demo = num_train_demo
        self.num_val_demo = num_val_demo
        self.shuffle = shuffle

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_statistics(self):
        raise NotImplementedError

    @abstractmethod
    def _read_h5df(self, h5df_file:str):
        raise NotImplementedError
    


