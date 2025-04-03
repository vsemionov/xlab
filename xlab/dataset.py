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

import copy
from typing import Optional
import warnings

import numpy as np
import torch
import torch.utils.data as data
import lightning as L
import datasets

from .tokenizer import Tokenizer
from .util import progress_bar, get_cache_dir


class TextDataset(data.Dataset):
    def __init__(
            self,
            path: str, name: Optional[str],
            splits: dict[str, float], split: str,
            tokenizer: Tokenizer,
            column: str = 'text',
            num_proc: int = 4, quiet: bool = False,
            progress: str = 'tqdm',
    ):
        super().__init__()
        self.quiet = quiet
        self.column = column
        self.num_proc = num_proc
        self.progress = progress
        self.tokenizer = tokenizer
        dataset = datasets.load_dataset(path, name, trust_remote_code=True)
        splits = self._split(dataset, splits)
        splits = {name: self._tokenize(split, tokenizer) for name, split in splits.items()}
        if not tokenizer.has_vocab():
            self._build_vocab(splits['train'], tokenizer)
        splits = {name: self._index(split, tokenizer) for name, split in splits.items()}
        self.dataset = splits[split].with_format('numpy', columns=['indices'], output_all_columns=True)

    def _split(self, dataset, splits):
        if isinstance(dataset, (datasets.DatasetDict, datasets.IterableDatasetDict)):
            dataset = datasets.concatenate_datasets(list(dataset.values()))
        total = len(dataset)
        results = {}
        for name, size in splits.items():
            split, dataset = dataset.train_test_split(train_size=int(size * total), seed=42).values()
            results[name] = split
        if not self.quiet:
            print(f'Splits: { {name: len(split) for name, split in results.items() } }')
            if len(dataset) > 0:
                warnings.warn(f'Unused samples: {len(dataset)} out of {total}')
        return results

    def _tokenize(self, dataset, tokenizer):
        def tokenize(row):
            row['tokens'] = tokenizer(row[column])
            return row
        column = self.column
        tokenizer = copy.copy(tokenizer).reset_vocab()  # discard vocabulary state to prevent cache misses
        dataset = dataset.map(tokenize, remove_columns=[column], num_proc=self.num_proc, desc='Tokenizing')
        return dataset

    def _build_vocab(self, dataset, tokenizer):
        batches = (sample['tokens'] for sample in progress_bar(dataset, kind=self.progress, desc='Building vocabulary'))
        tokenizer.build_vocab(batches)

    def _index(self, dataset, tokenizer):
        def index(row):
            row['indices'] = tokenizer.index(row['tokens'])
            return row
        dataset = dataset.map(index, remove_columns=['tokens'], num_proc=self.num_proc, desc='Indexing')
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]['indices']


class ChunkDataset(data.Dataset):
    def __init__(self, dataset: TextDataset, seq_len: int, progress: str = 'tqdm'):
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.progress = progress
        self.index = self._chunk(dataset)

    def _chunk(self, dataset):
        index = []
        for i, indices in enumerate(progress_bar(dataset, kind=self.progress, desc='Chunking')):
            n_samples = len(indices) + 1  # account for <sos>
            index.extend([(i, j) for j in range(n_samples)])
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset = self.dataset
        tokenizer = dataset.tokenizer
        window = self.seq_len + 1
        ds_idx, start_idx = self.index[idx]
        end_idx = start_idx + window
        indices = dataset[ds_idx]
        if start_idx > 0:
            start_idx -= 1
            end_idx -= 1
        else:
            sos_index = tokenizer[tokenizer.sos_token]
            indices = np.concatenate([[sos_index], indices[:window - 1]])
        if len(indices) < end_idx:
            eos_index = tokenizer[tokenizer.eos_token]
            pad_index = tokenizer[tokenizer.pad_token]
            padding_size = end_idx - len(indices)
            padding = np.array([pad_index]).repeat(padding_size)
            indices = np.concatenate([indices, [eos_index], padding])
        indices = indices[start_idx:end_idx]
        indices = torch.from_numpy(indices)
        x, y = indices[:-1], indices[1:]
        return x, y


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
            batch_size: int = 32, pin_memory: bool = False, num_workers: int = 4, persistent_workers: bool = False,
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
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.datasets = {}
        self._vocab_cache_path = get_cache_dir() / 'vocab.pt'

    def _text_dataset(self, split, **kwargs):
        return TextDataset(
            path=self.path, name=self.name,
            splits=self.splits, split=split,
            tokenizer=self.tokenizer,
            column=self.column,
            num_proc=self.num_proc,
            progress=self.progress,
            **kwargs
        )

    def _dataset(self, split, **kwargs):
        text_dataset = self._text_dataset(split, **kwargs)
        return ChunkDataset(text_dataset, seq_len=self.seq_len, progress=self.progress)

    def prepare_data(self):
        self._text_dataset('train')
        self.tokenizer.save_vocab(self._vocab_cache_path)

    def setup(self, stage):
        self.tokenizer.load_vocab(self._vocab_cache_path)
        kwargs = dict(quiet=True)
        if stage == 'fit':
            self.datasets['train'] = self._dataset('train', **kwargs)
            self.datasets['val'] = self._dataset('val', **kwargs)
        elif stage == 'validate':
            self.datasets['val'] = self._dataset('val', **kwargs)
        elif stage == 'test':
            self.datasets['test'] = self._dataset('test', **kwargs)
        elif stage == 'predict':
            self.datasets['predict'] = self._dataset('predict', **kwargs)
        else:
            assert False

    def train_dataloader(self):
        return data.DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
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
