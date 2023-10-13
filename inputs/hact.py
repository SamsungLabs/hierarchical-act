import glob
import math
import os
import random
from typing import List, Dict, Union

import gin
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from .base import BaseInput
from .registry import register

from core.data_format import DataStatistics
from languages.language_embedding import binary_encoder


@register('hact')
@gin.configurable(denylist=['data_root', 'split', 'shuffle'])
class HactInput(BaseInput):
    def __init__(self, 
                data_root:str,
                data_rel_dir:str,
                split:str='train',
                task_list: Union[str, List[str]]='all',
                num_train_demo: Union[str, int]='all',
                num_val_demo: Union[str, int]='all',
                shuffle:bool =True,
                chunk_size:int = 100,
                sentence_embedding_dim:int = 512,
                use_success_data_only:bool = False,
                image_keys:List[str]=['top']):
        self.data_dir = os.path.join(data_root, data_rel_dir)
        self.split = split
        self.task_list = task_list
        self.num_train_demo = num_train_demo
        self.num_val_demo = num_val_demo
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.sentence_embedding_dim = sentence_embedding_dim
        self.use_success_data_only = use_success_data_only
        self.image_keys = image_keys

    def __iter__(self):
        if self.use_success_data_only:
            file_list, description_list = self._get_h5df_list(domain='success')
        else:
            success_file_list, success_description_list = self._get_h5df_list(domain='success')
            failure_file_list, failure_description_list = self._get_h5df_list(domain='fail')
            file_list = success_file_list+failure_file_list
            description_list = success_description_list + failure_description_list

        if self.shuffle:
            temp = list(zip(file_list, description_list))
            random.shuffle(temp)
            file_list, description_list = zip(*temp)

        for data_file, description in zip(file_list, description_list):
            dataset = self._read_h5df(data_file, description)
            yield dataset

    def _get_h5df_list(self, domain='success'):
        file_list = []
        description_list = []
        if self.task_list == 'all':
            task_list = os.listdir(self.data_dir)
        else:
            task_list = self.task_list

        for task in task_list:
            h5path = os.path.join(self.data_dir, task,
                                    domain,
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

            description = self._task_to_description(task)
            description_list += [description]*len(h5files)
        return file_list, description_list
    
    def _task_to_description(self, task):
        #description = task.replace('_', ' ')
        description = task
        return description

    def _read_h5df(self, hdf5_file, description):
        with h5py.File(hdf5_file, 'r') as root:
            episode_len = root.attrs['episode_len']
            success = True if root.attrs['success'] =='success' else False
            #sentence_embedding = root['sentence_embedding'][:]
            sentence_embedding = binary_encoder(description, embedding_dim=self.sentence_embedding_dim)

            start_ts = np.random.randint(0, episode_len-1)
            end_ts = np.random.randint(start_ts+1, episode_len)
            time_cost = episode_len-start_ts
            
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

            images1 = []
            for key in self.image_keys:
                images1.append(root['images'][key][end_ts])
            images1 = np.stack(images1, axis=0) 

            actions_full = []
            for key in root['actions'].keys():
                actions_full.append(root['actions'][key][start_ts:end_ts+1])
            actions_full = np.concatenate(actions_full, axis=1)
            
            actions_inter, is_pad = self._compute_padding(actions_full, num_points=self.chunk_size)
            is_pad = torch.from_numpy(is_pad).bool()

            image0_data = torch.from_numpy(images0)
            image1_data = torch.from_numpy(images1)
            
            qpos_data = torch.from_numpy(qpos).float()
            actions_inter_data = torch.from_numpy(actions_inter).float()
            action_raw_data = torch.from_numpy(actions_full).float()
            context_data = torch.from_numpy(sentence_embedding).float()
            duration_data = torch.from_numpy(np.asarray(end_ts-start_ts)).float()
            

            image0_data = image0_data/255.0
            image0_data = torch.einsum('k h w c -> k c h w', image0_data)

            image1_data = image1_data/255.0
            image1_data = torch.einsum('k h w c -> k c h w', image1_data)
            
            context_data = context_data/255.0

            output_data = {
                'images': image0_data,
                'goal_images': image1_data,
                #'description': description,
                'contexts': context_data,
                'qpos': qpos_data,
                'actions': actions_inter_data,
                'duration': duration_data,
                'success': success,
                'is_pad': is_pad,
                'time_cost': time_cost,
            }
        return output_data

    def get_statistics(self):
        h5df_file_list, _ = self._get_h5df_list(domain='success')
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

    def _compute_padding(self, actions, num_points):
        actions = actions[:num_points]
        action_len = len(actions)

        original_action_shape = (num_points,)+actions.shape[1:]
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = actions
        padded_action[action_len:, :] = actions[-1,:]
        is_pad = np.zeros(num_points)
        is_pad[action_len:] = 1
        return padded_action, is_pad