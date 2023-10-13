import os

import gin
import numpy as np
import torch

import inputs
import models
import envs
from core.train_utils import CheckpointManager

@gin.configurable(denylist=['exp_dir'])
class Task():
    def __init__(self, 
                    exp_dir:str,
                    data_root:str,
                    env_id: str='mujoco_bimanipulation',
                    input_id: str='act',
                    model_id: str='act'):
        self.exp_dir = exp_dir
        self.data_root = data_root
        self.env_id = env_id
        self.model_id = model_id
        self.input_id = input_id

        self.ckpt_path = os.path.join(exp_dir, 'model.pt')
        self.checkpoint_manager = CheckpointManager(ckpt_path=self.ckpt_path)
        self.model = models.make(self.model_id)()
        self.env = envs.make(self.env_id)()

    def get_model(self, gpu):
        return self.model.cuda(gpu)

    def get_env(self):
        return self.env

    def get_dataset(self):
        train_ds = inputs.make(self.input_id)(
                    self.data_root,
                    split='train',
                    shuffle=True)
        val_ds = inputs.make(self.input_id)(
                    self.data_root,
                    split='val',
                    shuffle=False)
        return train_ds, val_ds
    

    def is_ckpt_exist(self):
        return os.path.exists(self.ckpt_path)

    def restore_model(self):
        self.model = self.checkpoint_manager.load(self.model)
    
    def save_model(self):
        self.checkpoint_manager.save(self.model)
    
    def load_training_ckpt(self, model, optimizer):
        model, optimizer, epoch, loss_dict \
                = self.checkpoint_manager.load(model, optimizer)
        self.model = model
        return model, optimizer, epoch, loss_dict
    
    def save_training_ckpt(self, 
                    model, 
                    optimizer,
                    epoch,
                    loss_dict
                    ):
        self.checkpoint_manager.save(model, 
                                     optimizer, 
                                     epoch, 
                                     loss_dict)