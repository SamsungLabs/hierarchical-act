import glob
import os
from abc import *
from typing import List, Dict, Tuple, Optional, Generator

import h5py
import numpy as np
import tqdm

from core.data_format import EpisodicDataClass
from languages import language_embedding as lang

class BaseDataProcessor():
    def __init__(self, 
                    source_dir: str, 
                    target_dir: str, 
                    file_format: str='',
                    language_embedding: str='binary_encoder',
                    language_embedding_dim: int=512):
        self.source_dir=source_dir
        self.target_dir=target_dir
        self.file_format=file_format
        self.language_embedding = language_embedding
        self.language_embedding_dim = language_embedding_dim

    def process_data(self):
        for target_file, episodic_data in self._generate_episodic_data():
            dir_name = os.path.dirname(target_file)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            self._write_file(target_file, episodic_data)
            
    def _get_embedding(self, description):
        if self.language_embedding == 'binary_encoder':
            return lang.binary_encoder(description,
                                       embedding_dim = self.language_embedding_dim)
        elif self.language_embedding =='universal_sentence_encoder':
            return lang.universe_sentence_encoder(description,
                                       embedding_dim = self.language_embedding_dim)
        else:
            raise Exception("Unsupported language embedding")

    def _write_file(self,
                    target_file: str,
                    episodic_data: EpisodicDataClass)->None:
        description = episodic_data.description
        sentence_embedding = episodic_data.sentence_embedding
        episode_len = episodic_data.episode_len
        qpos = episodic_data.qpos
        qvel = episodic_data.qvel
        images = episodic_data.images
        actions = episodic_data.actions
        
        with h5py.File(target_file, 'w') as root:
            root.attrs['description'] = description
            root.attrs['episode_len'] = episode_len

            root.create_dataset('sentence_embedding',
                                sentence_embedding.shape, dtype=np.float32)
            root['sentence_embedding'][...] = sentence_embedding

            qpos_group = root.create_group('qpos')
            qvel_group = root.create_group('qvel')
            images_group = root.create_group('images')
            actions_group = root.create_group('actions')

            for key, value in qpos.items():
                qpos_group.create_dataset(
                    key, value.shape, dtype=value.dtype
                )
                qpos_group[key][...] = value

            for key, value in qvel.items():
                qvel_group.create_dataset(
                    key, value.shape, dtype=value.dtype
                )
                qvel_group[key][...] = value

            for key, value in images.items():
                images_group.create_dataset(
                    key, value.shape, dtype=value.dtype
                )
                images_group[key][...] = value

            for key, value in actions.items():
                actions_group.create_dataset(
                    key, value.shape, dtype=value.dtype
                )
                actions_group[key][...] = value

    @abstractmethod
    def _generate_episodic_data(self)->Generator[Tuple[str, EpisodicDataClass], None, None]:
        raise NotImplementedError
