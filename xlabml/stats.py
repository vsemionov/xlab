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
from .datasets import TokenDataset
from .utils import is_ascii


def vocabulary_stats(tokenizer: Tokenizer):
    num_learned = sum(tokenizer.is_learned(index) for index in range(len(tokenizer)))
    char_indices = [index for index in range(len(tokenizer)) if tokenizer.is_char(index)]
    unicode_indices = [index for index in char_indices if not is_ascii(tokenizer.get_token(index))]
    return {
        'num_learned': num_learned,
        'num_chars': len(char_indices),
        'num_unicode': len(unicode_indices),
    }


def dataset_stats(dataset: TokenDataset, sample_size):
    np.random.seed(42)
    sample_indices = np.random.permutation(min(sample_size, len(dataset)))
    sample = dataset[sample_indices]
    lengths = np.array([len(indices) for indices in sample])
    return {
        'size_est': round(np.sum(lengths) * len(dataset) / len(sample)),
        'length_mean': np.mean(lengths),
        'length_median': np.median(lengths),
        'length_std': np.std(lengths),
    }


def compute_stats(tokenizer: Tokenizer, dataset: TokenDataset, sample_size: int = 10_000):
    return {
        'vocabulary': vocabulary_stats(tokenizer),
        'dataset': dataset_stats(dataset, sample_size),
    }
