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


class BaseDataset(data.Dataset):
    def __init__(self, column: str, parent: Union['BaseDataset', hf_datasets.Dataset], dataset: hf_datasets.Dataset):
        super().__init__()
        self.column = column
        self.parent = parent
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][self.column]

    def __iter__(self):
        for batch in self.dataset.iter(1000):
            for item in batch[self.column]:
                yield item


class TextDataset(BaseDataset):
    def __init__(
            self,
            locations: Union[HubLocation, dict, list[Union[HubLocation, dict]]],
            splits: dict[str, float], split: str,
            quiet: bool = False,
    ):
        if isinstance(locations, (HubLocation, dict)):
            locations = [locations]
        locations = [HubLocation(**location) if isinstance(location, dict) else location for location in locations]
        column = 'text'
        datasets = [hf_datasets.load_dataset(**location.to_load_kwargs()) for location in locations]
        datasets = [d.select_columns(l.column) if l.prune else d for l, d in zip(locations, datasets)]
        datasets = [d.rename_column(l.column, column) if l.column != column else d for l, d in zip(locations, datasets)]
        datasets = [d for ds in datasets for d in (ds.values() if isinstance(ds, dict) else [ds])]
        dataset = hf_datasets.concatenate_datasets(datasets)
        splits = self._split(dataset, splits, quiet)
        parent = splits[split]
        dataset = parent.select_columns(column)
        super().__init__(column=column, parent=parent, dataset=dataset)
        self.split = split

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


class TokenDataset(BaseDataset):
    def __init__(
            self,
            parent: TextDataset,
            tokenizer: Tokenizer,
            dynamic: bool = False,
            num_proc: int = 4,
    ):
        column = 'indices'
        dataset = self._encode(parent, tokenizer, column, dynamic, num_proc)
        super().__init__(column=column, parent=parent, dataset=dataset)
        self.tokenizer = tokenizer

    @staticmethod
    def _encode(parent, tokenizer, column, dynamic, num_proc):
        def encode(batch):
            return {column: tokenizer.encode(batch[parent.column])}
        def transform(batch):
            return {column: [np.array(indices) for indices in tokenizer.encode(batch[parent.column])]}

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


class SequenceDataset(BaseDataset):
    def __init__(
            self,
            parent: TokenDataset,
            seq_len: int, step_size: Union[float, int] = 0.5,
            num_proc: int = 4,
    ):
        step_size = int(step_size * seq_len) if isinstance(step_size, float) else step_size
        assert 0 < step_size <= seq_len
        column = 'index'
        dataset = self._index(parent, column, step_size, num_proc)
        super().__init__(column=column, parent=parent, dataset=dataset)
        self.seq_len = seq_len
        tokenizer = parent.tokenizer
        self.sos = np.array([tokenizer[tokenizer.sos_token]])
        self.eos = np.array([tokenizer[tokenizer.eos_token]])
        self.padding = np.array([tokenizer[tokenizer.pad_token]]).repeat(seq_len)

    @staticmethod
    def _index(parent, column, step_size, num_proc):
        def index(batch, idxs):
            return {
                column: [
                    (parent_idx, start_idx)
                    for parent_idx, indices in zip(idxs, batch[parent.column])
                    for start_idx in range(0, len(indices) + 1, step_size)  # +1 accounts for <sos>
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

    def __getitem__(self, idx):
        parent_idx, start_idx = self.dataset[idx][self.column]
        indices = self.parent[parent_idx]
        if start_idx > 0:
            indices = indices[start_idx - 1:start_idx + self.seq_len]
        else:
            indices = np.concatenate([self.sos, indices[:self.seq_len]])
        remainder = self.seq_len - len(indices) + 1
        if remainder > 0:
            indices = np.concatenate([indices, self.eos, self.padding[:remainder - 1]])
        indices = torch.from_numpy(indices)
        return indices[:-1], indices[1:]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
