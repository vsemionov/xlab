# Copyright 2025 Victor Semionov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import torch
import torch.utils.data as data
import lightning as L
from torchdata.stateful_dataloader import StatefulDataLoader

from .tokenizer import Tokenizer
from .datasets import TextDataset, ChunkDataset


class XLabDataModule(L.LightningDataModule):
    """XLab data module"""

    def __init__(
            self,
            path: str = 'wikipedia', name: Optional[str] = '20220301.simple',
            splits: dict[str, float] = {'train': 0.1, 'val': 0.05, 'test': 0.05, 'predict': 0.05},
            tokenizer: str = 'basic_english', language: str = 'en', max_tokens: int = 10_000,
            column: str = 'text',
            num_proc: int = 4,
            progress: str = 'tqdm',
            seq_len: int = 128,
            step_size: Union[float, int] = 0.5,
            batch_size: int = 128, pin_memory: bool = False, num_workers: int = 4, persistent_workers: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.path = path
        self.name = name
        self.splits = splits
        self.tokenizer = Tokenizer(tokenizer, language=language, max_tokens=max_tokens)
        self.column = column
        self.num_proc = num_proc
        self.progress = progress
        self.seq_len = seq_len
        self.step_size = step_size
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.datasets = {}

    def _dataset(self, split, **kwargs):
        text_dataset = TextDataset(
            path=self.path, name=self.name,
            splits=self.splits, split=split,
            tokenizer=self.tokenizer,
            column=self.column,
            num_proc=self.num_proc,
            progress=self.progress,
            **kwargs
        )
        chunk_dataset = ChunkDataset(
            text_dataset,
            seq_len=self.seq_len, step_size=self.step_size,
            num_proc=self.num_proc,
            progress=self.progress,
        )
        return chunk_dataset

    def prepare_data(self):
        self.datasets['train'] = self._dataset('train')
        for split in ['val', 'test', 'predict']:
            self.datasets[split] = self._dataset(split, quiet=True)

    def setup(self, stage):
        splits = {
            'fit': ['train', 'val'],
            'validate': ['val'],
            'test': ['test'],
            'predict': ['predict'],
        }
        self.datasets = {split: dataset for split, dataset in self.datasets.items() if split in splits[stage]}
        for split in splits[stage]:
            if split not in self.datasets:
                self.datasets[split] = self._dataset(split, quiet=True)

    def train_dataloader(self):
        return StatefulDataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            generator=torch.Generator().manual_seed(42),
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.datasets['predict'],
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def state_dict(self):
        return {'vocab': self.tokenizer.vocab}

    def load_state_dict(self, state_dict):
        self.tokenizer.vocab = state_dict['vocab']
