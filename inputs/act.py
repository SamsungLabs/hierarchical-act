import glob
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

@register('act')
@gin.configurable(denylist=['data_root', 'split', 'shuffle'])
class ActInput(BaseInput):
    def __init__(self, 
                data_root:str,
                data_rel_dir:str,
                split:str='train',
                task_list: Union[str, List[str]]='all',
                num_train_demo: Union[str, int]='all',
                num_val_demo: Union[str, int]='all',
                shuffle:bool =True,
                chunk_size:int = 100,
                sample_full_episode:bool = False):
        self.data_dir = os.path.join(data_root, data_rel_dir)
        self.split = split
        self.task_list = task_list
        self.num_train_demo = num_train_demo
        self.num_val_demo = num_val_demo
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.sample_full_episode = sample_full_episode

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
            h5path = os.path.join(self.data_dir, task, 
                                    self.split, '*.hdf5')
            h5files = sorted(glob.glob(h5path))

            if self.split == 'train' and self.num_train_demo != 'all':
                h5files = h5files[:self.num_train_demo]
            elif self.split =='val'and self.num_val_demo != 'all':
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
            qpos_keys = sorted(list(root['qpos'].keys())) 
            for key in qpos_keys:
                qpos.append(root['qpos'][key][start_ts])
            qpos = np.concatenate(qpos, axis=0)

            qvel = []
            qvel_keys = sorted(list(root['qvel'].keys())) 
            for key in qvel_keys:
                qvel.append(root['qvel'][key][start_ts])
            qvel = np.concatenate(qvel, axis=0)

            images = []
            images_keys = sorted(list(root['images'].keys()))
            for key in images_keys:
                images.append(root['images'][key][start_ts])
            images = np.stack(images, axis=0) 

            actions_full = []
            actions_keys = sorted(list(root['actions'].keys()))
            for key in actions_keys:
                actions_full.append(root['actions'][key][start_ts:start_ts+self.chunk_size])
            actions_full = np.concatenate(actions_full, axis=1)
            action_len = actions_full.shape[0]

            original_action_shape = (self.chunk_size,)+actions_full.shape[1:]
            padded_action = np.zeros(original_action_shape, dtype=np.float32)
            padded_action[:action_len] = actions_full
            padded_action[action_len:] = actions_full[-1]
            is_pad = np.zeros(self.chunk_size)
            is_pad[action_len:] = 1

            image_data = torch.from_numpy(images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()
            context_data = torch.from_numpy(sentence_embedding).float()

            image_data = image_data/255.0
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            context_data = context_data/255.0
            
            output_data = {
                'images': image_data,
                'contexts': context_data,
                'qpos': qpos_data,
                'actions': action_data,
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
