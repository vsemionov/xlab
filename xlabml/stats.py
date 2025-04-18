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

import numpy as np

from .tokenizer import Tokenizer
from .datasets import SequenceDataset
from .text import is_ascii


def vocabulary_stats(tokenizer: Tokenizer):
    num_learned = sum(tokenizer.is_learned(index) for index in range(len(tokenizer)))
    char_indices = [index for index in range(len(tokenizer)) if tokenizer.is_char(index)]
    unicode_indices = [index for index in char_indices if not is_ascii(tokenizer.get_token(index))]
    return {
        'num_learned': num_learned,
        'num_chars': len(char_indices),
        'num_unicode': len(unicode_indices),
    }


def dataset_stats(dataset: SequenceDataset, sample_size: int):
    np.random.seed(42)
    token_dataset = dataset.parent
    text_dataset = token_dataset.parent
    tokenizer = token_dataset.tokenizer
    pad_index = tokenizer[tokenizer.pad_token]
    text_sample_size = min(sample_size, len(token_dataset))
    chunk_sample_size = min(sample_size, len(dataset))
    text_sample_indices = np.random.permutation(text_sample_size)
    chunk_sample_indices = np.random.permutation(chunk_sample_size)
    text_sample = text_dataset[text_sample_indices]
    token_sample = token_dataset[text_sample_indices]
    chunk_sample = [dataset[int(idx)][0] for idx in chunk_sample_indices]
    text_lengths = np.array([len(text) for text in text_sample])
    token_lengths = np.array([len(indices) for indices in token_sample])
    return {
        'text_size_est': round(np.sum(text_lengths) * len(token_dataset) / text_sample_size),
        'token_size_est': round(np.sum(token_lengths) * len(token_dataset) / text_sample_size),
        'text_length_mean': np.mean(text_lengths),
        'text_length_median': np.median(text_lengths),
        'text_length_std': np.std(text_lengths),
        'token_length_mean': np.mean(token_lengths),
        'token_length_median': np.median(token_lengths),
        'token_length_std': np.std(token_lengths),
        'seq_fill_ratio_mean': np.mean([(chunk != pad_index).sum() for chunk in chunk_sample]) / dataset.seq_len,
    }


def compute_stats(
        tokenizer: Tokenizer,
        dataset: SequenceDataset,
        sample_size: int = 10_000
):
    return {
        'vocabulary': vocabulary_stats(tokenizer),
        'dataset': dataset_stats(dataset, sample_size),
    }
