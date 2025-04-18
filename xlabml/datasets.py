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
import warnings

import numpy as np
import torch
import torch.utils.data as data
import datasets

from .tokenizer import Tokenizer
from . import DATA_DIR


class TextDataset(data.Dataset):
    def __init__(
            self,
            path: str, name: Optional[str],
            splits: dict[str, float], split: str,
            column: str = 'text',
            save_splits: bool = False,
            num_proc: int = 4,
            quiet: bool = False,
    ):
        super().__init__()
        dataset_dir = DATA_DIR / path / (name or '')
        split_dir = dataset_dir / split
        source = None
        if save_splits:
            try:
                source = datasets.load_from_disk(split_dir)
            except FileNotFoundError:
                pass
        if source is None:
            dataset = datasets.load_dataset(path, name, trust_remote_code=True)
            splits = self._split(dataset, splits, quiet)
            if save_splits:
                splits.save_to_disk(dataset_dir, num_proc=num_proc)
                source = datasets.load_from_disk(split_dir)  # reload prevents cache miss downstream
            else:
                source = splits[split]
        self.column = column
        self.parent = None
        self.source = source
        self.dataset = source.select_columns([column])

    @staticmethod
    def _split(dataset, splits, quiet):
        if isinstance(dataset, dict):
            dataset = datasets.concatenate_datasets(list(dataset.values()))
        total = len(dataset)
        results = datasets.DatasetDict()
        for name, size in splits.items():
            if size > 0:
                size = int(size * total) if isinstance(size, float) else size
                split, dataset = dataset.train_test_split(train_size=size, shuffle=True, seed=42).values()
            else:
                split, dataset = dataset, []
            results[name] = split
        if not quiet:
            print(f'Splits: { {name: len(split) for name, split in results.items()} }')
            if len(dataset) > 0:
                warnings.warn(f'Unused samples: {len(dataset)} out of {total}')
        return results

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][self.column]


class TokenDataset(data.Dataset):
    def __init__(
            self,
            parent: TextDataset,
            tokenizer: Tokenizer,
            dynamic: bool = False,
            num_proc: int = 4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.column = 'indices'
        self.parent = parent
        self.source = parent.dataset
        self.dataset = self._encode(self.parent, tokenizer, dynamic, num_proc)

    def _encode(self, parent, tokenizer, dynamic, num_proc):
        def encode(batch):
            return {self.column: tokenizer.encode(batch[parent.column])}
        def transform(batch):
            return {self.column: [np.array(indices) for indices in tokenizer.encode(batch[parent.column])]}

        if dynamic:
            return parent.dataset.with_transform(transform)
        else:
            return parent.dataset.map(
                encode,
                batched=True,
                remove_columns=parent.dataset.column_names,
                num_proc=num_proc,
                desc='Encoding',
            ).with_format('numpy')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][self.column]


class ChunkDataset(data.Dataset):
    def __init__(
            self,
            parent: TokenDataset,
            seq_len: int, step_size: Union[float, int] = 0.5,
            num_proc: int = 4,
    ):
        super().__init__()
        step_size = int(step_size * seq_len) if isinstance(step_size, float) else step_size
        assert 0 < step_size <= seq_len
        self.seq_len = seq_len
        self.step_size = step_size
        self.column = 'index'
        self.parent = parent
        self.source = parent.dataset
        self.dataset = self._index(parent, num_proc)

    def _index(self, parent, num_proc):
        def index(batch, ds_idxs):
            return {
                self.column: [
                    (parent_idx, start_idx)
                    for parent_idx, indices in zip(ds_idxs, batch[parent.column])
                    for start_idx in range(0, len(indices) + 1, self.step_size)  # +1 accounts for <sos>
                ]
            }

        return parent.dataset.map(
            index,
            with_indices=True,
            batched=True,
            remove_columns=parent.dataset.column_names,
            num_proc=num_proc,
            desc='Indexing',
        ).with_format()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        parent = self.parent
        tokenizer = parent.tokenizer
        window = self.seq_len + 1
        parent_idx, start_idx = self.dataset[idx][self.column]
        end_idx = start_idx + window
        indices = parent[parent_idx]
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
