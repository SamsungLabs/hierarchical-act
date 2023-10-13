import abc
import os
from typing import List, Optional

import gin
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

#from apex.parallel import DistributedDataParallel
from apex import amp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from common import nested_tensor_utils
from core.train_utils import EpochManager, CheckpointManager
from core import Task
from optimizers import TorchOptimizerBuilder


class Trainer():
    def __init__(self, 
                 task:Task,
                 train_batch_size: int=8,
                 val_batch_size: int=1,
                 train_epoch: int=1992,
                 val_epoch_interval: int=3,
                 gpus: List[int]=[0],
                 seed: Optional[int]=None):
        self.task = task
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_epoch = train_epoch
        self.gpus = gpus
        self.seed = seed
        self.exp_dir = self.task.exp_dir
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.log_file = os.path.join(self.exp_dir, 'loss.log')
        self.epoch_manager = EpochManager(log_path=self.log_file)
        
    @abc.abstractmethod
    def build(self, gpu):
        pass
    
    @abc.abstractmethod
    def train_one_step(self, data):
        pass

    @abc.abstractmethod
    def val_one_step(self, data):
        pass
    
    @abc.abstractmethod
    def train(self):
        pass

    def train_on_single_gpu(self, gpu):
        torch.cuda.set_device(gpu)
        model, optimizer, train_ds, val_ds = self.build(gpu)
        
        train_dataloader = DataLoader(
            train_ds,
            batch_size = self.train_batch_size,
            pin_memory=True, 
            num_workers=1, 
            prefetch_factor=1
        )

        val_dataloader = DataLoader(
            val_ds,
            batch_size = self.val_batch_size,
            pin_memory=True, 
            num_workers=1, 
            prefetch_factor=1
        )
        
        if self.task.is_ckpt_exist():
            model, optimizer, saved_epoch, losses = self.task.load_training_ckpt(model, optimizer)
            init_loss = losses['total_loss']
            print(f'ckpt epoch: {saved_epoch}, ckpt total_loss: {init_loss:.6f}')
            start_epoch = saved_epoch+1
        else:
            start_epoch = 0

        train_total = 0
        val_total = 0
        for epoch in range(start_epoch, self.train_epoch):
            model.train()
            with self.epoch_manager as em:
                if epoch == start_epoch:
                    pbar = tqdm(train_dataloader)
                else:
                    pbar = tqdm(train_dataloader, total=train_total)

                for data in pbar:
                    one_step_losses = self.train_one_step(data, model, optimizer, gpu)
                    
                    if gpu == self.gpus[0]:
                        em.update(one_step_losses)

                        prefix = f'[Train epoch {epoch}] '
                        log = em.get_log(prefix)
                        pbar.set_description(log)
                        
                        if epoch == start_epoch:
                            train_total +=1

                losses = em.get_avg_losses()

            if epoch%self.val_epoch_interval==0:
                with torch.inference_mode():
                    model.eval()
                    with self.epoch_manager as em:
                        if epoch == start_epoch:
                            pbar = tqdm(val_dataloader)
                        else:
                            pbar = tqdm(val_dataloader, total=val_total)

                        for data in pbar:
                            one_step_losses = self.val_one_step(data, model, optimizer, gpu)

                            if gpu == self.gpus[0]:
                                em.update(one_step_losses)
                                prefix = f'[Val epoch {epoch}] '
                                log = em.get_log(prefix)
                                pbar.set_description(log)

                                if epoch == start_epoch:
                                    val_total +=1

                self.task.save_training_ckpt(model, optimizer, epoch, losses)


@gin.configurable(denylist=['task', 'gpu'])
class SingleGpuTrainer(Trainer):
    def __init__(self, 
                    task:Task,
                    gpu: int=0,
                    train_batch_size: int=8,
                    val_batch_size: int=1,
                    train_epoch: int=1992,
                    val_epoch_interval: int=3,
                    seed: Optional[int]=None):
        super(SingleGpuTrainer, self).__init__(task,
                                               train_batch_size=train_batch_size,
                                               val_batch_size=val_batch_size,
                                               train_epoch=train_epoch,
                                               val_epoch_interval=val_epoch_interval,
                                               gpus=[gpu],
                                               seed=seed)
        self.task = task
        self.gpus = [gpu]
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_epoch = train_epoch
        self.val_epoch_interval = val_epoch_interval
        self.seed = seed
    
    def build(self, gpu):
        model = self.task.get_model(gpu)
        train_ds, val_ds = self.task.get_dataset()   
        model.set_data_statistics(train_ds.get_statistics())
        
        optimizer_builder = TorchOptimizerBuilder()
        optimizer = optimizer_builder.build(model.parameters())
        return model, optimizer, train_ds, val_ds
    
    def train_one_step(self, data, model, optimizer, gpu):
        optimizer.zero_grad()
        data = nested_tensor_utils.to_device(data, 
                                            device=gpu,
                                            non_blocking=True)
        loss, loss_output = model.compute_train_loss(data)
        loss.backward()
        optimizer.step()
        return loss_output
    
    def val_one_step(self, data, model, optimizer, gpu):
        data = nested_tensor_utils.to_device(data, 
                                             device=gpu,
                                             non_blocking=True)
        with torch.inference_mode():
            loss, loss_output = model.compute_val_loss(data)
        return loss_output

    def train(self):
        self.train_on_single_gpu(self.gpus[0])
    
    
@gin.configurable(denylist=['task'])
class MultiGpuTrainer(Trainer):
    def __init__(self, 
                    task:Task,
                    train_batch_size: int=8,
                    val_batch_size: int=1,
                    train_epoch: int=1992,
                    val_epoch_interval: int=3,
                    
                    gpus: List[int]=[0,1,2,3],
                    num_nodes: int=1,
                    ranking_with_nodes: int=0,
                    num_workers: int=8,
                    seed: Optional[int]=None):
        super(MultiGpuTrainer, self).__init__(task,
                                               train_batch_size=train_batch_size,
                                               val_batch_size=val_batch_size,
                                               train_epoch=train_epoch,
                                               val_epoch_interval=val_epoch_interval,
                                               gpus=gpus,
                                               seed=seed)
        self.task = task
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_epoch = train_epoch
        self.val_epoch_interval = val_epoch_interval
        self.seed = seed
        
        self.num_nodes = num_nodes
        self.ranking_with_nodes = ranking_with_nodes
        self.num_workers = num_workers
        self.gpus = gpus
        self.master_addr = '127.0.0.1'
        self.master_port = '2202'
        
    def build(self, gpu):
        rank = self.ranking_with_nodes * len(self.gpus) + gpu
        world_size = self.num_nodes * len(self.gpus)
        print('world_size', world_size)
        print('rank', rank)
        ## gin issues
        
        dist.init_process_group(backend='nccl', 
                                init_method=f'tcp://{self.master_addr}:{self.master_port}', 
                                world_size=world_size, rank=rank)
        
    
        model = self.task.get_model(gpu)
        train_ds, val_ds = self.task.get_dataset()
        model.set_data_statistics(train_ds.get_statistics())
        
        optimizer_builder = TorchOptimizerBuilder()
        optimizer = optimizer_builder.build(model.parameters())
        model, optimizer = amp.initialize(model, optimizer,
                                      opt_level='O1')
        model = DistributedDataParallel(model, 
                                        device_ids=[gpu],
                                        output_device=gpu)
        return model, optimizer, train_ds, val_ds
    

    def train_one_step(self, data, model, optimizer, gpu):
        optimizer.zero_grad()
        data = nested_tensor_utils.to_device(data, 
                                            device=gpu,
                                            non_blocking=True)
        loss, loss_output = model.module.compute_train_loss(data)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        return loss_output
    
    
    def val_one_step(self, data, model, optimizer, gpu):
        data = nested_tensor_utils.to_device(data, 
                                             device=gpu,
                                             non_blocking=True)
        with torch.inference_mode():
            loss, loss_output = model.module.compute_val_loss(data)
        return loss_output
    
    def train(self):
        os.environ['MASTER_ADDR'] = self.master_addr
        os.environ['MASTER_PORT'] = self.master_port
        mp.spawn(self.train_on_single_gpu, 
                 nprocs=len(self.gpus),
                 args=())