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

from dataclasses import dataclass, asdict
from typing import Optional, Union
import warnings

import numpy as np
import torch
import torch.utils.data as data
import datasets as hf_datasets

from .tokenizer import Tokenizer


@dataclass
class HubLocation:
    path: str
    name: Optional[str] = None
    split: Optional[str] = None
    trust_remote_code: bool = False
    column: str = 'text'
    prune: bool = False

    def to_load_kwargs(self):
        exclude = {'column', 'prune'}
        return {k: v for k, v in asdict(self).items() if k not in exclude}


class TextDataset(data.Dataset):
    def __init__(
            self,
            locations: Union[HubLocation, list[HubLocation]],
            splits: dict[str, float], split: str,
            quiet: bool = False,
    ):
        super().__init__()
        if isinstance(locations, HubLocation):
            locations = [locations]
        column = 'text'
        datasets = [hf_datasets.load_dataset(**location.to_load_kwargs()) for location in locations]
        datasets = [d.select_columns(l.column) if l.prune else d for l, d in zip(locations, datasets)]
        datasets = [d.rename_column(l.column, column) if l.column != column else d for l, d in zip(locations, datasets)]
        datasets = [d for ds in datasets for d in (ds.values() if isinstance(ds, dict) else [ds])]
        dataset = hf_datasets.concatenate_datasets(datasets)
        splits = self._split(dataset, splits, quiet)
        source = splits[split]
        self.column = column
        self.parent = None
        self.source = source
        self.dataset = source.select_columns([self.column])

    @staticmethod
    def _split(dataset, splits, quiet):
        total = len(dataset)
        results = hf_datasets.DatasetDict()
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


class SequenceDataset(data.Dataset):
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
