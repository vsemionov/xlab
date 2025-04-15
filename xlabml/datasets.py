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
from joblib import Parallel, delayed

from .tokenizer import Tokenizer


class TextDataset(data.Dataset):
    def __init__(
            self,
            path: str, name: Optional[str],
            splits: dict[str, float], split: str,
            column: str = 'text',
            bulk_options: Optional[dict] = None,
            quiet: bool = False,
    ):
        super().__init__()
        self.column = column
        dataset = datasets.load_dataset(path, name, trust_remote_code=True)
        splits = self._split(dataset, splits, bulk_options, quiet)
        self.parent = splits[split]
        self.dataset = self.parent.select_columns([self.column])

    @staticmethod
    def _split(dataset, splits, bulk_options, quiet):
        if isinstance(dataset, dict):
            dataset = datasets.concatenate_datasets(list(dataset.values()))
        total = len(dataset)
        results = {}
        for name, size in splits.items():
            if size > 0:
                size = int(size * total) if isinstance(size, float) else size
                split, dataset = dataset.train_test_split(
                    train_size=size,
                    shuffle=True,
                    seed=42,
                    **(bulk_options or {})
                ).values()
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
            dataset: TextDataset,
            tokenizer: Tokenizer,
            dynamic: bool = False,
            bulk_options: Optional[dict] = None,
    ):
        super().__init__()
        self.column = 'indices'
        self.parent = dataset
        self.tokenizer = tokenizer
        self.dataset = self._encode(self.parent, tokenizer, dynamic, bulk_options)

    def _encode(self, dataset, tokenizer, dynamic, bulk_options):
        def encode(batch):
            return {self.column: tokenizer.encode(batch[dataset.column])}
        def transform(batch):
            return {self.column: [np.array(indices) for indices in tokenizer.encode(batch[dataset.column])]}

        if dynamic:
            return dataset.dataset.with_transform(transform)
        else:
            return dataset.dataset.map(
                encode,
                batched=True,
                remove_columns=dataset.dataset.column_names,
                desc='Encoding',
                **(bulk_options or {})
            ).with_format('numpy')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][self.column]


class ChunkDataset(data.Dataset):
    def __init__(
            self,
            dataset: TokenDataset,
            seq_len: int, step_size: Union[float, int] = 0.5,
            bulk_options: Optional[dict] = None,
            progress: str = 'tqdm',
    ):
        super().__init__()
        self.column = 'chunks'
        self.dataset = dataset
        self.seq_len = seq_len
        self.step_size = int(step_size * seq_len) if isinstance(step_size, float) else step_size
        assert 0 < self.step_size <= self.seq_len
        self.progress = progress
        self.index = self._index(self.dataset, bulk_options)

    def _index(self, dataset, bulk_options):
        def index(batch, ds_idxs):
            return {
                self.column: [
                    (ds_idx, start_idx)
                    for ds_idx, indices in zip(ds_idxs, batch[dataset.column])
                    for start_idx in range(0, len(indices) + 1, self.step_size)  # +1 accounts for <sos>
                ]
            }

        return dataset.dataset.map(
            index,
            with_indices=True,
            batched=True,
            remove_columns=dataset.dataset.column_names,
            desc='Indexing',
            **(bulk_options or {})
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset = self.dataset
        tokenizer = dataset.tokenizer
        window = self.seq_len + 1
        ds_idx, start_idx = self.index[idx][self.column]
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


def parallelize(dataset, batch_size=1000, n_jobs=1, threaded=False):
    prefer = 'threads' if threaded else 'processes'
    parallel = Parallel(n_jobs=n_jobs, return_as='generator', prefer=prefer)
    batches = parallel(
        delayed(dataset.__getitem__)(slice(start, start + batch_size))
        for start in range(0, len(dataset), batch_size)
    )
    samples = (sample for batch in batches for sample in batch)
    return samples
