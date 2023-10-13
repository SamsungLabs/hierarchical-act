import glob
import os
import json

import IPython
import h5py
from typing import List, Dict, Union

import numpy as np
import tqdm

from .base import BaseDataProcessor
from .registry import register
from core.data_format import DomainEpisodicDataClass

@register('srrc_dual_frankas')
class SrrcDualFrankasDataProcessor(BaseDataProcessor):
    def __init__(self, 
                    source_dir:str,
                    target_dir:str,
                    split_train_ratio: float = 0.7,
                    skip_param: int=1,
                    file_format='*.npy',
                    language_embedding: str='binary_encoder',
                    language_embedding_dim: int=512):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.file_format = file_format
        self.split_train_ratio = split_train_ratio
        self.language_embedding = language_embedding
        self.language_embedding_dim = language_embedding_dim
        self.skip_param = skip_param
        self.meta_info_name = 'meta_data.json'
        self.cri = None

    def _generate_episodic_data(self):
        source_folders = self._get_source_folder_list()
        for idx, source_folder in tqdm.tqdm(enumerate(source_folders)):
            source_files = self._get_source_file_list(source_folder)
            episodic_data = self._read_file(source_files)
            target_file = source_folder.replace(self.source_dir,
                                                os.path.join(self.target_dir, os.path.basename(self.source_dir)))
            target_file = target_file.replace('success/', '')[:-1] + '.hdf5'
            head, tail = os.path.split(target_file)
            if idx >= self.cri:
                target_file = os.path.join(head, 'val')
            else:
                target_file = os.path.join(head, 'train')
            target_file = os.path.join(target_file, tail)
            yield target_file, episodic_data

    def _get_source_folder_list(self):
        folder_path = os.path.join(self.source_dir, 'success', 'episode*/')
        folder_list = glob.glob(folder_path, recursive=True)
        if not folder_list:
            folder_path = os.path.join(self.source_dir, 'episode*/')
            folder_list = glob.glob(folder_path, recursive=True)
        folder_list = sorted(folder_list)
        self.cri = int(len(folder_list) * self.split_train_ratio)
        return folder_list

    def _get_source_file_list(self, target_folder):
        file_path = os.path.join(target_folder, '**', self.file_format)
        file_list = glob.glob(file_path, recursive = True)
        file_list = sorted(file_list)
        return file_list[::self.skip_param]

    def _read_file(self, npy_files)->DomainEpisodicDataClass:
        with open(os.path.join(os.path.dirname(npy_files[0]), self.meta_info_name)) as f:
            meta_info = json.load(f)
        description = meta_info['description']
        sentence_embedding = self._get_embedding(description)

        qpos_dict = dict()
        qpos_dict['left_arm_qpos'] = np.empty((0, 7), dtype=float)
        qpos_dict['left_gripper_qpos'] = np.empty((0, 1), dtype=float)
        qpos_dict['right_arm_qpos'] = np.empty((0, 7), dtype=float)
        qpos_dict['right_gripper_qpos'] = np.empty((0, 1), dtype=float)

        qvel_dict = dict()
        qvel_dict['left_arm_qvel'] = np.empty((0, 7), dtype=float)
        qvel_dict['left_gripper_qvel'] = np.empty((0, 1), dtype=float)
        qvel_dict['right_arm_qvel'] = np.empty((0, 7), dtype=float)
        qvel_dict['right_gripper_qvel'] = np.empty((0, 1), dtype=float)

        root = np.load(npy_files[0], allow_pickle=True).item()
        img_dict = dict()
        for key in root['obs'].keys():
            if 'images' in key:
                img_dict[key.replace('images/', '')] = np.empty((0, 480, 640, 3), dtype=np.uint8)
        act_dict = dict()
        act_dict['left_arm_qpos'] = np.empty((0, 7), dtype=float)
        act_dict['left_gripper_qpos'] = np.empty((0, 1), dtype=float)
        act_dict['right_arm_qpos'] = np.empty((0, 7), dtype=float)
        act_dict['right_gripper_qpos'] = np.empty((0, 1), dtype=float)
        
        for npy_file in npy_files:
            print('process:', npy_file)
            root = np.load(npy_file, allow_pickle=True).item()
            left_arm_qpos = np.expand_dims(root['obs']['joint_position/left_arm'], axis=0)
            qpos_dict['left_arm_qpos'] = np.append(qpos_dict['left_arm_qpos'], left_arm_qpos, axis=0)
            left_gripper_qpos = np.expand_dims(root['obs']['joint_position/left_hand'], axis=0)
            qpos_dict['left_gripper_qpos'] = np.append(qpos_dict['left_gripper_qpos'], left_gripper_qpos, axis=0)
            right_arm_qpos = np.expand_dims(root['obs']['joint_position/right_arm'], axis=0)
            qpos_dict['right_arm_qpos'] = np.append(qpos_dict['right_arm_qpos'], right_arm_qpos, axis=0)
            right_gripper_qpos = np.expand_dims(root['obs']['joint_position/right_hand'], axis=0)
            qpos_dict['right_gripper_qpos'] = np.append(qpos_dict['right_gripper_qpos'], right_gripper_qpos, axis=0)

            left_arm_qvel = np.expand_dims(root['obs']['joint_velocity/left_arm'], axis=0)
            qvel_dict['left_arm_qvel'] = np.append(qvel_dict['left_arm_qvel'], left_arm_qvel, axis=0)
            left_gripper_qvel = np.expand_dims(root['obs']['joint_velocity/left_hand'], axis=0)
            qvel_dict['left_gripper_qvel'] = np.append(qvel_dict['left_gripper_qvel'], left_gripper_qvel, axis=0)
            right_arm_qvel = np.expand_dims(root['obs']['joint_velocity/right_arm'], axis=0)
            qvel_dict['right_arm_qvel'] = np.append(qvel_dict['right_arm_qvel'], right_arm_qvel, axis=0)
            right_gripper_qvel = np.expand_dims(root['obs']['joint_velocity/right_hand'], axis=0)
            qvel_dict['right_gripper_qvel'] = np.append(qvel_dict['right_gripper_qvel'], right_gripper_qvel, axis=0)

            # iterable, right_head, right_hand, left_head, left_hand
            for key in root['obs'].keys():
                if 'images' in key:
                    _key = key.replace('images/', '')
                    img = np.expand_dims(root['obs'][key], axis=0)
                    img_dict[_key] = np.append(img_dict[_key], img, axis=0)
            
            action_left_arm_qpos = np.expand_dims(root['action']['joint_command/left_arm'], axis=0)
            act_dict['left_arm_qpos'] = np.append(act_dict['left_arm_qpos'], action_left_arm_qpos, axis=0)
            action_left_gripper_qpos = np.expand_dims(root['action']['joint_command/left_hand'], axis=0)
            act_dict['left_gripper_qpos'] = np.append(act_dict['left_gripper_qpos'], action_left_gripper_qpos, axis=0)
            action_right_arm_qpos = np.expand_dims(root['action']['joint_command/right_arm'], axis=0)
            act_dict['right_arm_qpos'] = np.append(act_dict['right_arm_qpos'], action_right_arm_qpos, axis=0)
            action_right_gripper_qpos = np.expand_dims(root['action']['joint_command/right_hand'], axis=0)
            act_dict['right_gripper_qpos'] = np.append(act_dict['right_gripper_qpos'], action_right_gripper_qpos, axis=0)
            #import IPython; IPython.embed()
        episode_len = len(act_dict['left_arm_qpos'])

        episodic_data = DomainEpisodicDataClass(
            description=description,
            domain = '',
            sentence_embedding=sentence_embedding,
            success=True,
            episode_len=episode_len,
            qpos=qpos_dict,
            qvel=qvel_dict,
            images=img_dict,
            actions=act_dict
        )
        return episodic_data
