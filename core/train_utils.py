import copy
import os
from enum import Enum
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type:str='average'):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type == 'none':
            fmtstr = ''
        elif self.summary_type == 'average':
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type == 'sum':
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type == 'count':
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
 


class EpochManager():
    def __init__(self, 
                 log_path: str=None):
        self.log_path = log_path

    def __enter__(self):
        self._step = 0
        self._last_log = ''
        self._loss_dict = dict()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_path:
            self.write_log(self._last_log+"\n")
        self._step = 0
        self._last_log = ''
        self._loss_dict = dict()

    def update(self, one_step_loss_dict: Dict[str, float]):
        for key, value in one_step_loss_dict.items():
            if key in self._loss_dict:
                i = self._step
                p_loss = self._loss_dict[key]
                c_loss = value.detach().cpu().numpy()
                loss = (p_loss*(i+1) + c_loss)/(i+2)
            else:
                loss = value.detach().cpu().numpy()
            self._loss_dict.update({
                key: loss
            })
        self._step+=1
    
    def get_log(self, prefix :str=''):
        log = copy.copy(prefix)
        for key, value in self._loss_dict.items():
            log += f'{key}: {value:.6f} '
        self._last_log = copy.copy(log)
        return log

    def write_log(self, log:str):
        with open(self.log_path, 'a') as f:
            f.write(log)

    def get_avg_losses(self):
        return self._loss_dict



class CheckpointManager():
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self._last_loss = np.inf

    def save(self, model, optimizer, epoch, loss_dict):
        loss = loss_dict['total_loss']
        if loss < self._last_loss:
            self._last_loss = loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_dict
                }, self.ckpt_path
            )

    def load(self, model, optimizer=None, device="cuda"):
        checkpoint = torch.load(self.ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device(device)
        model.to(device)
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss_dict = checkpoint['loss']
            self._last_loss = loss_dict['total_loss']
            return model, optimizer, epoch, loss_dict
        else:
            return model
