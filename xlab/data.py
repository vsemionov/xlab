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

from functools import partial
from typing import Optional, Callable
import warnings

import numpy as np
import torch
import torch.utils.data as data
import lightning as L
import torchtext
import datasets
import platformdirs
from tqdm.auto import tqdm

from . import config


class TokenDataset(data.Dataset):
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    unk_token = '<unk>'
    specials = [pad_token, sos_token, eos_token, unk_token]

    def __init__(self,
            path: str, name: Optional[str],
            tokenizer: Callable[[str], list[str]], max_tokens: int,
            splits: dict[str, float], split: str,
            vocab: Optional[torchtext.vocab.Vocab] = None,
            num_proc: int = 4, quiet: bool = False,
    ):
        super().__init__()
        self.num_proc = num_proc
        self.quiet = quiet
        dataset = datasets.load_dataset(path, name, trust_remote_code=True)
        dataset = self._tokenize(dataset, tokenizer)
        splits = self._split(dataset, splits)
        self.vocab = self._index(splits['train'], max_tokens) if vocab is None else vocab
        self.dataset = self._vectorize(splits[split], self.vocab)

    def _tokenize(self, dataset, tokenizer):
        def tokenize(row):
            row['tokens'] = tokenizer(row['text'])
            return row
        dataset = dataset.map(tokenize, remove_columns=['text'], num_proc=self.num_proc, desc='Tokenizing')
        return dataset

    def _split(self, dataset, splits):
        if isinstance(dataset, (datasets.DatasetDict, datasets.IterableDatasetDict)):
            dataset = datasets.concatenate_datasets(list(dataset.values()))
        total = len(dataset)
        results = {}
        for name, size in splits.items():
            split, dataset = dataset.train_test_split(train_size=int(size * total), seed=0).values()
            results[name] = split
        if not self.quiet:
            print(f'Splits: { {name: len(split) for name, split in results.items() } }')
        if len(dataset) > 0:
            warnings.warn(f'Unused samples: {len(dataset)} out of {total}')
        return results

    def _index(self, dataset, max_tokens):
        iterator = (sample['tokens'] for sample in tqdm(dataset, desc='Indexing'))
        vocab = torchtext.vocab.build_vocab_from_iterator(iterator, specials=self.specials, max_tokens=max_tokens)
        vocab.set_default_index(self.specials.index('<unk>'))
        return vocab

    def _vectorize(self, dataset, vocab):
        def vectorize(row):
            row['indices'] = np.array(vocab.lookup_indices(row['tokens']))
            return row
        dataset = dataset.map(vectorize, remove_columns=['tokens'], num_proc=self.num_proc, desc='Vectorizing')
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]['indices']


class SequenceDataset(data.Dataset):
    def __init__(self, dataset: TokenDataset, seq_len: int):
        self.dataset = dataset
        self.seq_len = seq_len
        self.index = self._chunk(dataset)

    def _chunk(self, dataset):
        index = []
        for i, indices in enumerate(tqdm(dataset, desc='Chunking')):
            n_samples = len(indices) + 1  # account for <sos>
            index.extend([(i, j) for j in range(n_samples)])
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset = self.dataset
        window = self.seq_len + 1
        ds_idx, start_idx = self.index[idx]
        end_idx = start_idx + window
        indices = dataset[ds_idx]
        if start_idx > 0:
            start_idx -= 1
            end_idx -= 1
        else:
            sos_index = dataset.vocab[dataset.sos_token]
            indices = np.concatenate([[sos_index], indices[:window - 1]])
        if len(indices) < end_idx:
            eos_index = dataset.vocab[dataset.eos_token]
            pad_index = dataset.vocab[dataset.pad_token]
            padding_size = max(end_idx - len(indices) - 1, 0)
            padding = np.array([pad_index]).repeat(padding_size)
            indices = np.concatenate([indices, [eos_index], padding])
        indices = indices[start_idx:end_idx]
        indices = torch.from_numpy(indices)
        x, y = indices[:-1], indices[1:]
        return x, y


class XLabDataset(L.LightningDataModule):
    def __init__(self,
            path: str = 'wikipedia', name: Optional[str] = '20220301.simple',
            tokenizer: str = 'basic_english', max_tokens: int = 20_000,
            splits: dict[str, float] = {'train': 0.1, 'val': 0.05, 'test': 0.05, 'predict': 0.05},
            seq_len: int = 128,
            batch_size: int = 32, num_workers: int = 4, persistent_workers: bool = True,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.tokenizer = torchtext.data.utils.get_tokenizer(tokenizer)
        self.max_tokens = max_tokens
        self.splits = splits
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.datasets = {}
        self._td_init = partial(
            TokenDataset,
            path=self.path, name=self.name,
            tokenizer=self.tokenizer, max_tokens=self.max_tokens,
            splits=self.splits,
            num_proc=self.num_workers,
        )
        self._cache_dir = platformdirs.user_cache_path(config.APP_NAME, ensure_exists=True)

    def prepare_data(self):
        token_dataset = self._td_init(split='train')
        torch.save(token_dataset.vocab, self._cache_dir / 'vocab.pt')

    def setup(self, stage):
        vocab = torch.load(self._cache_dir / 'vocab.pt')
        td_init = partial(self._td_init, vocab=vocab, quiet=True)
        dataset_init = partial(SequenceDataset, seq_len=self.seq_len)
        if stage == 'fit':
            self.datasets['train'] = dataset_init(td_init(split='train'))
            self.datasets['val'] = dataset_init(td_init(split='val'))
        elif stage == 'validate':
            self.datasets['val'] = dataset_init(td_init(split='val'))
        elif stage == 'test':
            self.datasets['test'] = dataset_init(td_init(split='test'))
        elif stage == 'predict':
            self.datasets['predict'] = dataset_init(td_init(split='predict'))
        else:
            assert False

    def train_dataloader(self):
        return data.DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return data.DataLoader(
            self.datasets['predict'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
