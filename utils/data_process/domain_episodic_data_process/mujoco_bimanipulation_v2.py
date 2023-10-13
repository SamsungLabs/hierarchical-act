import glob
import os
import h5py
from typing import List, Dict, Union

import gin
import numpy as np
import tqdm

from .base import BaseDataProcessor
from .registry import register
from core.data_format import DomainEpisodicDataClass

@register('mujoco_bimanipulation_v2')
class MujocoBimanipulationV2DataProcessor(BaseDataProcessor):
    def __init__(self, 
                    source_dir:str,
                    target_dir:str,
                    file_format='episode_*.hdf5',
                    language_embedding: str='binary_encoder',
                    language_embedding_dim: int=512):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.file_format = file_format
        self.language_embedding = language_embedding
        self.language_embedding_dim = language_embedding_dim
        
    def _generate_episodic_data(self):
        for sf in ['success', 'fail']:
            source_files = self._get_source_file_list(sf=sf)
            for source_file in tqdm.tqdm(source_files):
                episodic_data =  self._read_file(source_file, sf=sf)
                target_file = source_file.replace(self.source_dir, self.target_dir)
                target_file = os.path.splitext(target_file)[0] + '.hdf5'
                yield target_file, episodic_data


    def _get_source_file_list(self, sf='success'):
        file_path = os.path.join(self.source_dir, '**', sf, '**', self.file_format)
        file_list = glob.glob(file_path, recursive = True)
        file_list = sorted(file_list)
        return file_list

    def _read_file(self, 
                   hdf5_file,
                   sf='success')->List[DomainEpisodicDataClass]:
        with h5py.File(hdf5_file, 'r') as root:
            #to do: description = root.attrs['description']
            description = ''
            sentence_embedding = self._get_embedding(description)

            qpos_dict = dict()
            qpos_dict['left_arm_qpos'] = root['/observations/qpos'][:][:, 0:6]
            qpos_dict['left_gripper_positions'] = root['/observations/qpos'][:][:, 6:7]
            qpos_dict['right_arm_qpos'] = root['/observations/qpos'][:][:, 7:13]
            qpos_dict['right_gripper_positions'] = root['/observations/qpos'][:][:, 13:14]
            
            qvel_dict = dict()
            qvel_dict['left_arm_qvel'] = root['/observations/qvel'][:][:, 0:6]
            qvel_dict['left_gripper_velocity'] = root['/observations/qvel'][:][:, 6:7]
            qvel_dict['right_arm_qvel'] = root['/observations/qvel'][:][:, 7:13]
            qvel_dict['right_gripper_velocity'] = root['/observations/qvel'][:][:, 13:14]

            img_dict = dict()
            for key, value in root[f'/observations/images'].items():
                img_dict[f'/images/{key}'] = value[:]
            act_dict = dict()
            act_dict['left_arm_qpos'] = root['/action'][:][:, 0:6]
            act_dict['left_gripper_positions'] = root['/action'][:][:, 6:7]
            act_dict['right_arm_qpos'] = root['/action'][:][:, 7:13]
            act_dict['right_gripper_positions'] = root['/action'][:][:, 13:14]

            episode_len = len(act_dict['left_arm_qpos'])
            

        episodic_data = DomainEpisodicDataClass(
            description=description,
            domain = '',
            sentence_embedding=sentence_embedding,
            success=True if sf=='success' else False,
            episode_len=episode_len,
            qpos=qpos_dict,
            qvel=qvel_dict,
            images=img_dict,
            actions=act_dict
        )
        return episodic_data
