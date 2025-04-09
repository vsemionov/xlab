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
import datasets
from joblib import Parallel, delayed

from .tokenizer import Tokenizer
from .util import progress_bar, cached


def fingerprint(dataset):
    return dataset._fingerprint


def parallelize(dataset, column=None, n_jobs=1, threaded=False):
    batch_size = 1000
    prefer = 'threads' if threaded else 'processes'
    parallel = Parallel(n_jobs=n_jobs, return_as='generator', prefer=prefer)
    batches = parallel(
        delayed(dataset.__getitem__)(slice(start, start + batch_size))
        for start in range(0, len(dataset), batch_size)
    )
    samples = (sample for batch in batches for sample in (batch[column] if column is not None else batch))
    return samples


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
        samples = parallelize(dataset, column='tokens', n_jobs=self.num_proc)
        batches = progress_bar(samples, kind=self.progress, total=len(dataset), desc='Building vocabulary')
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
    def __init__(
            self,
            dataset: TextDataset,
            seq_len: int, step_size: Union[float, int] = 0.5,
            num_proc: int = 4,
            progress: str = 'tqdm'
    ):
        super().__init__()
        self.dataset = dataset
        self.seq_len = seq_len
        self.step_size = int(step_size * seq_len) if isinstance(step_size, float) else step_size
        assert 0 < self.step_size <= self.seq_len
        self.num_proc = num_proc
        self.progress = progress
        self.index = cached(lambda: self._chunk(dataset), 'index', fingerprint(dataset.dataset))

    def _chunk(self, dataset):
        index = []
        samples = parallelize(dataset, n_jobs=self.num_proc, threaded=True)
        for i, indices in enumerate(progress_bar(samples, kind=self.progress, total=len(dataset), desc='Chunking')):
            index.extend([(i, j) for j in range(0, len(indices) + 1, self.step_size)])  # 1 accounts for <sos>
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
