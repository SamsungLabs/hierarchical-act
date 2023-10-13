import glob
import os
import h5py
from typing import List, Dict

import numpy as np
import math
import torch
import tqdm

from .base import BaseDataProcessor
from .registry import register
from core.data_format import EpisodicDataClass

@register('maniskill2')
class Maniskill2DataProcessor(BaseDataProcessor):
    def __init__(self,
                    source_dir:str,
                    target_dir:str,
                    split_train_ratio:float=0.7,
                    file_format='*.h5',
                    language_embedding: str='binary_encoder',
                    language_embedding_dim: int=512):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.file_format = file_format
        self.split_train_ratio = split_train_ratio
        self.language_embedding = language_embedding
        self.language_embedding_dim = language_embedding_dim
        self._episode_num = None
        self._cri = None
        self._index_unit = 1000

    def _generate_episodic_data(self):
        source_files = self._get_source_file_list()
        for source_file in source_files:
            print(f'{source_file} processing...')
            total_episodes = self._get_episodic_data_len(source_file)
            self._cri = int(total_episodes * self.split_train_ratio)
            self._episode_num = 0
            for idx in range(math.ceil(total_episodes / self._index_unit)):
                episodic_data_list = self._read_file(hdf5_file=source_file, idx_unit=idx)
                for episodic_data in tqdm.tqdm(episodic_data_list, desc=f'writing...{idx}'):
                    target_file = source_file.replace(self.source_dir, self.target_dir)

                    if self._episode_num >= self._cri:
                        target_file = os.path.join(os.path.dirname(target_file), 'val')
                    else:
                        target_file = os.path.join(os.path.dirname(target_file), 'train')
                    _, tail = os.path.split(source_file)
                    target_file = os.path.join(target_file, tail)
                    target_file = os.path.splitext(target_file)[0] + f'_{self._episode_num}' '.hdf5'
                    dir_name = os.path.dirname(target_file)
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    self._episode_num += 1
                    yield target_file, episodic_data

    def _get_source_file_list(self):
        file_path = os.path.join(self.source_dir, '**', self.file_format)
        file_list = glob.glob(file_path, recursive=True)
        file_list = sorted(file_list)
        return file_list

    def _get_episodic_data_len(self, hdf5_file)->int:
        with h5py.File(hdf5_file, 'r') as root:
            return len(root.keys())

    def _read_file(self, hdf5_file, idx_unit:int)->List[EpisodicDataClass]:
        '''
        qpos    : (9,)
        images  : (128, 128, 3)
        actions : (8,)
        '''

        episodic_data = []
        with h5py.File(hdf5_file, 'r') as root:
            key_list = sorted(list(root.keys()))
            for episode in tqdm.tqdm(key_list[idx_unit * self._index_unit:(idx_unit + 1) * self._index_unit],
                                     desc=f'reading...{idx_unit}'):
                description = root[f'{episode}'].attrs['description']
                sentence_embedding = self._get_embedding(description)
                qpos_dict = dict()
                qpos_dict['qpos'] = root[f'{episode}/obs/agent/qpos'][:-1]

                qvel_dict = dict()
                qvel_dict['qvel'] = root[f'{episode}/obs/agent/qvel'][:-1]

                img_dict = dict()
                for key, value in root[f'{episode}/obs/image'].items():
                    img_dict[f'/images/{key}'] = value['rgb'][:-1]

                act_dict = dict()
                act_dict['qpos'] = root[f'{episode}/actions'][:]

                episode_len = len(act_dict['qpos'])

                episodic_data.append(
                    EpisodicDataClass(
                    description=description,
                    sentence_embedding=sentence_embedding,
                    episode_len=episode_len,
                    qpos=qpos_dict,
                    qvel=qvel_dict,
                    images=img_dict,
                    actions=act_dict
                    )
                )
        return episodic_data
