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

from pathlib import Path
from typing import Optional, Union, Iterable
import warnings

import torch
import torch.utils.data as data
import lightning as L
from torchdata.stateful_dataloader import StatefulDataLoader
from boltons.setutils import IndexedSet

from .tokenizer import Tokenizer, TokenizerTrainer
from .datasets import TextDataset, TokenDataset, ChunkDataset, parallelize
from .utils import download


class XLabDataModule(L.LightningDataModule):
    """XLab data module"""

    def __init__(
            self,
            path: str = 'wikipedia', name: Optional[str] = '20220301.simple',
            splits: dict[str, float] = {'train': 0.1, 'val': 0.05, 'test': 0.05, 'predict': 0.05},  # noqa
            column: str = 'text',
            num_tokens: int = 10_000,
            tokenizer_url: Optional[str] = None,
            tokenizer_path: Path = Path('tokenizers/default.tok'),
            tokenizer_train_args: dict = TokenizerTrainer().train_args,
            bulk_options: Optional[dict[str, dict]] = None,
            dynamic_encode: bool = False,
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
        self.column = column
        self.num_tokens = num_tokens
        self.tokenizer_url = tokenizer_url
        self.tokenizer_path = tokenizer_path
        self.tokenizer_trainer = TokenizerTrainer(tokenizer_train_args)
        self.tokenizer: Optional[Tokenizer] = None
        self.bulk_options = self._get_bulk_options(bulk_options)
        self.dynamic_encode = dynamic_encode
        self.progress = progress
        self.seq_len = seq_len
        self.step_size = step_size
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.datasets = {}

    def _get_bulk_options(self, options):
        options = options or {}
        bulk_options = {
            'split': {
                'writer_batch_size': 1000,
                **options.get('split', {})
            },
            'encode': {
                'batch_size': 1000,
                'num_proc': 4,
                **options.get('encode', {})
            },
            'index': {
                'batch_size': 1000,
                'num_proc': 4,
                **options.get('index', {})
            },
            'tokenizer_train_load': {
                'batch_size': 1000,
                'n_jobs': 4,
                'threaded': False,
                **options.get('tokenizer_train_load', {})
            },
        }
        return bulk_options

    def _create_tokenizer(self, dataset):
        try:
            return Tokenizer.load(self.tokenizer_path)
        except FileNotFoundError:
            pass
        if self.tokenizer_url:
            download(self.tokenizer_url, self.tokenizer_path)
            return Tokenizer.load(self.tokenizer_path)
        else:
            texts = parallelize(dataset, **self.bulk_options['tokenizer_train_load'])
            return self.tokenizer_trainer.train(texts, self.num_tokens, self.tokenizer_path)

    def create_datasets_and_tokenizer(self, splits: Optional[Iterable[str]] = None, level: Optional[str] = None):
        splits = splits if splits is not None else list(self.splits)
        assert 'train' in splits or self.tokenizer is not None

        text_datasets = {
            split: TextDataset(
                path=self.path,
                name=self.name,
                splits=self.splits,
                split=split,
                column=self.column,
                bulk_options=self.bulk_options['split'],
                quiet=(i != 0),
            )
            for i, split in enumerate(splits)
        }
        if level == 'text':
            return text_datasets

        if self.tokenizer is None:
            self.tokenizer = self._create_tokenizer(text_datasets['train'])
            vocab_size = len(self.tokenizer)
            num_tokens = self.num_tokens
            if vocab_size < num_tokens:
                warnings.warn(
                    f'Tokenizer vocabulary has size {vocab_size}, which is less than the configured {num_tokens}. '
                    f'Model dimensions are linked to the configured size, which is incorrect.'
                )
            if vocab_size > num_tokens:
                raise ValueError(
                    f'Tokenizer vocabulary has size {vocab_size}, which is more than the configured {num_tokens}. '
                    f'Model dimensions are linked to the configured size, which is incorrect.'
                )
        if level == 'tokenizer':
            return self.tokenizer

        token_datasets = {
            split: TokenDataset(
                dataset=text_dataset,
                tokenizer=self.tokenizer,
                dynamic=self.dynamic_encode,
                bulk_options=self.bulk_options['encode'],
            )
            for split, text_dataset in text_datasets.items()
        }
        chunk_datasets = {
            split: ChunkDataset(
                dataset=token_dataset,
                seq_len=self.seq_len, step_size=self.step_size,
                bulk_options=self.bulk_options['index'],
                progress=self.progress,
            )
            for split, token_dataset in token_datasets.items()
        }
        return chunk_datasets

    def prepare_data(self):
        self.datasets = self.create_datasets_and_tokenizer()

    def setup(self, stage):
        splits = {
            'fit': ['train', 'val'],
            'validate': ['val'],
            'test': ['test'],
            'predict': ['predict'],
        }
        self.datasets = {split: dataset for split, dataset in self.datasets.items() if split in splits[stage]}
        new_splits = IndexedSet(splits[stage]) - self.datasets.keys()
        self.datasets |= self.create_datasets_and_tokenizer(splits=new_splits)

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
