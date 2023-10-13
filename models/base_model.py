from abc import *

import torch

class BaseModel(torch.nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def set_data_statistics(self):
        raise NotImplementedError

    @abstractmethod
    def compute_train_loss(self, data, optimizer):
        raise NotImplementedError
    
    @abstractmethod
    def compute_val_loss(self, data):
        raise NotImplementedError