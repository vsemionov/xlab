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
            locations: Union[HubLocation, list[HubLocation]],
            splits: dict[str, float], split: str,
            quiet: bool = False,
    ):
        if isinstance(locations, HubLocation):
            locations = [locations]
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
    parent: TokenDataset

    def __init__(
            self,
            parent: TokenDataset,
            seq_len: int, step_size: Union[float, int] = 0.5,
            concatenate: bool = False, pad_incomplete: bool = True,
            train_sos: bool = False,
            num_proc: int = 4,
    ):
        step_size = int(step_size * seq_len) if isinstance(step_size, float) else step_size
        assert 0 < step_size <= seq_len
        if not concatenate and not pad_incomplete:
            warnings.warn(
                'Sequence concatenation and padding are both disabled. The model will see very few <eos> tokens.'
            )
        column = 'indices'
        tokenizer = parent.tokenizer
        sos_index = tokenizer[tokenizer.sos_token]
        eos_index = tokenizer[tokenizer.eos_token]
        pad_index = tokenizer[tokenizer.pad_token]
        dataset = self._generate(parent, column, seq_len, step_size, concatenate, pad_incomplete, train_sos, num_proc,
            sos_index, eos_index, pad_index)
        super().__init__(column=column, parent=parent, dataset=dataset)
        self.concatenate = concatenate
        self.train_sos = train_sos
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.pad_index = pad_index

    @staticmethod
    def _generate(parent, column, seq_len, step_size, concatenate, pad_incomplete, train_sos, num_proc,
            sos_index, eos_index, pad_index):
        # When train_sos is true and padding is enabled, sos will be added after eos, before regular padding,
        # thus training the model to start a new sequence after finishing a previous one.
        def generate():
            sos = np.array([sos_index])
            eos = np.array([eos_index])
            padding = np.array([pad_index]).repeat(seq_len)
            sos_pad = np.array([sos_index])

            reader = iter(parent)
            buffer = np.array([], dtype=int)
            window = seq_len + 1
            add_sos = train_sos

            while True:
                if len(buffer) < window:
                    if concatenate:
                        try:
                            buffer = np.concatenate([buffer, sos, next(reader), eos])
                            continue
                        except StopIteration:
                            pass

                    buf_thresh = 1 - add_sos  # require trainable tokens in buffer (1 if training sos, 2 otherwise)
                    # can add sos if pad=true or exactly token left
                    if (pad_incomplete or len(buffer) == window - add_sos) \
                            and len(buffer) > buf_thresh and buffer[buf_thresh] != pad_index:
                        buffer = np.concatenate([buffer, sos_pad[:add_sos], padding[:window - len(buffer) - add_sos]])
                        add_sos = False  # disable until next read
                    else:
                        try:
                            buffer = np.concatenate([sos, next(reader), eos])
                            add_sos = train_sos  # reset
                            continue
                        except StopIteration:
                            break

                yield {column: buffer[:window]}
                buffer = buffer[step_size:]

        dataset = hf_datasets.Dataset.from_generator(generate, num_proc=num_proc, split=parent.parent.split)
        return dataset.with_format('numpy')

    def _compute_mask(self, x: torch.Tensor):
        sos_indices = (x == self.sos_index).nonzero().squeeze(1)
        if sos_indices.size(0) == 0:
            return torch.ones((x.size(0),) * 2, dtype=torch.bool).tril()
        lengths = sos_indices[1:] - sos_indices[:1]
        init = sos_indices[:1]
        remainder = x.size(0) - sos_indices[-1:]
        lengths = torch.cat([init, lengths, remainder])
        blocks = [torch.ones(l, l, dtype=torch.bool).tril() for l in lengths if l]
        return torch.block_diag(*blocks)

    def _get_xym(self, indices):
        indices = torch.from_numpy(indices)
        x, y = indices[:-1], indices[1:]
        if self.concatenate and not self.train_sos:  # unwanted target sos may be present only if concatenate is enabled
            y = torch.where(y == self.sos_index, self.pad_index, y)
        return (x, y, self._compute_mask(x)) if self.concatenate else (x, y)

    def __getitem__(self, idx):
        indices = self.dataset[idx][self.column]
        return self._get_xym(indices)


    def __iter__(self):
        for batch in self.dataset.iter(1000):
            for indices in batch[self.column]:
                yield self._get_xym(indices)
