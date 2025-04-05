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
from typing import Optional, Union
import warnings

import numpy as np
import torch
import torch.utils.data as data
import lightning as L
import datasets

from .tokenizer import Tokenizer
from .util import progress_bar, cached, fingerprint


class TextDataset(data.Dataset):
    def __init__(
            self,
            path: str, name: Optional[str],
            splits: dict[str, float], split: str,
            tokenizer: Tokenizer,
            column: str = 'text',
            num_proc: int = 4, quiet: bool = False,
            progress: str = 'tqdm',
            keep_text: bool = False,
            keep_tokens: bool = False,
    ):
        super().__init__()
        self.quiet = quiet
        self.column = column
        self.num_proc = num_proc
        self.progress = progress
        self.tokenizer = tokenizer
        self.keep_text = keep_text
        self.keep_tokens = keep_tokens
        dataset = datasets.load_dataset(path, name, trust_remote_code=True)
        splits = self._split(dataset, splits)
        splits = {name: self._tokenize(split, tokenizer) for name, split in splits.items()}
        if tokenizer.vocab is None:
            train_set = splits['train']
            tokenizer.vocab = cached(lambda: self._build_vocab(train_set, tokenizer), 'vocab', fingerprint(train_set))
        splits = {name: self._index(split, tokenizer) for name, split in splits.items()}
        self.dataset = splits[split].with_format('numpy', columns=['indices'], output_all_columns=True)

    def _split(self, dataset, splits):
        if isinstance(dataset, (datasets.DatasetDict, datasets.IterableDatasetDict)):
            dataset = datasets.concatenate_datasets(list(dataset.values()))
        total = len(dataset)
        results = {}
        for name, size in splits.items():
            if size > 0:
                size = int(size * total) if isinstance(size, float) else size
                split, dataset = dataset.train_test_split(train_size=size, seed=42).values()
            else:
                split, dataset = dataset, []
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
        remove_columns = [] if self.keep_text else [column]
        tokenizer = copy.copy(tokenizer).reset_vocab()  # discard vocabulary state to prevent cache misses
        dataset = dataset.map(tokenize, remove_columns=remove_columns, num_proc=self.num_proc, desc='Tokenizing')
        return dataset

    def _build_vocab(self, dataset, tokenizer):
        batches = (sample['tokens'] for sample in progress_bar(dataset, kind=self.progress, desc='Building vocabulary'))
        return tokenizer.build_vocab(batches).vocab

    def _index(self, dataset, tokenizer):
        def index(row):
            row['indices'] = tokenizer.index(row['tokens'])
            return row
        remove_columns = [] if self.keep_tokens else ['tokens']
        dataset = dataset.map(index, remove_columns=remove_columns, num_proc=self.num_proc, desc='Indexing')
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]['indices']


class ChunkDataset(data.Dataset):
    def __init__(self, dataset: TextDataset, seq_len: int, chunk_size: Union[float, int] = 0.5, progress: str = 'tqdm'):
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.chunk_size = int(chunk_size * seq_len) if isinstance(chunk_size, float) else chunk_size
        assert 0 < self.chunk_size <= self.seq_len
        self.progress = progress
        self.index = cached(lambda: self._chunk(dataset), 'index', fingerprint(dataset.dataset))

    def _chunk(self, dataset):
        index = []
        for i, indices in enumerate(progress_bar(dataset, kind=self.progress, desc='Chunking')):
            # integer arithmetic equivalent of math.ceil((len(indices) + 1) / self.chunk_size)  # 1 accounts for <sos>
            n_chunks = (len(indices) + self.chunk_size) // self.chunk_size
            index.extend([(i, j * self.chunk_size) for j in range(n_chunks)])
        # use smaller dtypes to save memory; can be further optimized by using a separate array for the small 2nd index
        dtype = np.uint32 if len(dataset) < 2**32 else np.uint64
        index = np.array(index, dtype=dtype)
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset = self.dataset
        tokenizer = dataset.tokenizer
        window = self.seq_len + 1
        ds_idx, start_idx = self.index[idx].tolist()
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
            chunk_size: Union[float, int] = 0.5,
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
        self.chunk_size = chunk_size
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
            seq_len=self.seq_len, chunk_size=self.chunk_size, progress=self.progress,
        )
        return chunk_dataset

    def _load_dataset(self, split, **kwargs):
        if split not in self.datasets:
            self.datasets[split] = self._dataset(split, **kwargs)

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
        for split in splits[stage]:
            self._load_dataset(split, quiet=True)

    def train_dataloader(self):
        return data.DataLoader(
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
