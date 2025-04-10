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

import re
from typing import Optional, Iterable
import warnings

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator

class Tokenizer:
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    unk_token = '<unk>'
    specials = [pad_token, sos_token, eos_token, unk_token]

    def __init__(self, tokenizer: str, language: str = 'en', max_tokens: int = 10_000):
        self.tokenizer = get_tokenizer(tokenizer, language=language)
        self.max_tokens = max_tokens
        self.vocab: Optional[Vocab] = None

    def _escape(self, text):
        return re.sub(rf"({'|'.join(self.specials)})", r'#\1', text)

    def _unescape(self, text):
        return re.sub(rf"#({'|'.join(self.specials)})", r'\1', text)

    def tokenize(self, text: str) -> list[str]:
        return self.tokenizer(self._escape(text))

    def detokenize(self, tokens: list[str]) -> str:
        return self._unescape(' '.join(tokens))

    def index(self, tokens: list[str]) -> list[int]:
        return self.vocab.lookup_indices(tokens)

    def deindex(self, indices: list[int]) -> list[str]:
        return self.vocab.lookup_tokens(indices)

    def encode(self, text: str) -> list[int]:
        return self.index(self.tokenize(text))

    def decode(self, indices: list[int]) -> str:
        return self.detokenize(self.deindex(indices))

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]

    def build_vocab(self, batches: Iterable[list[str]]) -> 'Tokenizer':
        self.vocab = build_vocab_from_iterator(batches, specials=self.specials, max_tokens=self.max_tokens)
        self.vocab.set_default_index(self.vocab[self.unk_token])
        if len(self.vocab) < self.max_tokens:
            warnings.warn(
                f'Built vocabulary has size {len(self.vocab)}, which is less than the maximum {self.max_tokens}. '
                f'Model dimensions are linked to the maximum size, which is incorrect.'
            )
        return self

    def reset_vocab(self) -> 'Tokenizer':
        self.vocab = None
        return self
