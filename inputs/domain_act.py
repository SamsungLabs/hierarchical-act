import glob
import math
import os
import random
from typing import List, Dict, Union

import gin
import h5py
import numpy as np
import torch
from .base import BaseInput
from .registry import register

from core.data_format import DataStatistics

@register('domain_act')
@gin.configurable(denylist=['data_root', 'split', 'shuffle'])
class DomainActInput(BaseInput):
    def __init__(self, 
                data_root:str,
                data_rel_dir:str,
                split:str='train',
                task_list: Union[str, List[str]]='all',
                num_train_demo: Union[str, int]='all',
                num_val_demo: Union[str, int]='all',
                shuffle:bool =True,
                chunk_size:int = 100,
                sample_full_episode:bool = False,
                image_keys:List[str] = ['top']):
        self.data_dir = os.path.join(data_root, data_rel_dir)
        self.split = split
        self.task_list = task_list
        self.num_train_demo = num_train_demo
        self.num_val_demo = num_val_demo
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.sample_full_episode = sample_full_episode
        self.image_keys = image_keys

    def __iter__(self):
        h5df_file_list = self._get_h5df_list()
        if self.shuffle:
            random.shuffle(h5df_file_list)

        for h5df_file in h5df_file_list:
            data_output = self._read_h5df(h5df_file)
            yield data_output


    def _get_h5df_list(self):
        file_list = []
        if self.task_list == 'all':
            task_list = os.listdir(self.data_dir)
        else:
            task_list = self.task_list

        for task in task_list:
            h5path = os.path.join(self.data_dir, task, 'success',
                                    self.split, '*.hdf5')
            h5files = sorted(glob.glob(h5path))

            if self.split == 'train' and self.num_train_demo != 'all':
                skip = max(int(len(h5files)/self.num_train_demo), 1)
                h5files = h5files[::skip]
                h5files = h5files[:self.num_train_demo]
            elif self.split =='val'and self.num_val_demo != 'all':
                skip = max(int(len(h5files)/self.num_val_demo), 1)
                h5files = h5files[::skip]
                h5files = h5files[:self.num_val_demo]
            else:
                h5files = h5files

            if len(h5files) == 0:
                raise Exception(f"Data not exist: {h5path}")
            file_list += h5files
        return file_list
    
    def _read_h5df(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as root:
            episode_len = root.attrs['episode_len']
            sentence_embedding = root['sentence_embedding'][:]

            if self.sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            qpos = []
            for key in root['qpos'].keys():
                qpos.append(root['qpos'][key][start_ts])
            qpos = np.concatenate(qpos, axis=0)

            qvel = []
            for key in root['qvel'].keys():
                qvel.append(root['qvel'][key][start_ts])
            qvel = np.concatenate(qvel, axis=0)

            images0 = []
            for key in self.image_keys:
                images0.append(root['images'][key][start_ts])
            images0 = np.stack(images0, axis=0) 

            actions_full = []
            for key in root['actions'].keys():
                actions_full.append(root['actions'][key][start_ts:start_ts+self.chunk_size:1])
            actions_full = np.concatenate(actions_full, axis=1)
            
            action_len = actions_full.shape[0]
            original_action_shape = (self.chunk_size,)+actions_full.shape[1:]
            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = actions_full
            is_pad = np.zeros(self.chunk_size)
            is_pad[action_len:] = 1

            image0_data = torch.from_numpy(images0)
            qpos_data = torch.from_numpy(qpos).float()
            actions_data = torch.from_numpy(padded_action).float()
            context_data = torch.from_numpy(sentence_embedding).float()
            is_pad = torch.from_numpy(is_pad).bool()

            image0_data = image0_data/255.0
            image0_data = torch.einsum('k h w c -> k c h w', image0_data)

            context_data = context_data/255.0
          
            output_data = {
                'images': image0_data,
                'contexts': context_data,
                'qpos': qpos_data,
                'actions': actions_data,
                'is_pad': is_pad,
            }
        return output_data


    def get_statistics(self):
        h5df_file_list = self._get_h5df_list()
        if len(h5df_file_list)> 50:
            h5df_file_list = random.sample(h5df_file_list, 50)

        all_qpos = []
        all_action = []
        for h5df_file in h5df_file_list:
            with h5py.File(h5df_file, 'r') as root:
                qpos = []
                for key in root['qpos'].keys():
                    qpos.append(root['qpos'][key][:])
                qpos = np.concatenate(qpos, axis=1)

                action = []
                for key in root['actions'].keys():
                    action.append(root['actions'][key][:])
                action = np.concatenate(action, axis=1)

                all_qpos.append(qpos)
                all_action.append(action)

        all_qpos = np.concatenate(all_qpos, axis=0)
        all_action = np.concatenate(all_action, axis=0)
        
        action_mean = np.mean(all_action, axis=0, keepdims=True)
        action_std = np.std(all_action, axis=0, keepdims=True)
        action_std = np.clip(action_std, 1e-2, 10) # clipping

        # normalize qpos data
        qpos_mean = np.mean(all_qpos, axis=0, keepdims=True)
        qpos_std = np.std(all_qpos, axis=0, keepdims=True)
        qpos_std = np.clip(qpos_std, 1e-2, 10) # clipping

        stats = DataStatistics(
            is_sim = True,
            action_mean =action_mean.squeeze(),
            action_std=action_std.squeeze(),
            qpos_mean=qpos_mean.squeeze(),
            qpos_std=qpos_std.squeeze(),
        )
        return stats
