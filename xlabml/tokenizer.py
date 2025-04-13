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
from pathlib import Path
from typing import Union, Iterable, Callable

import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator

class Tokenizer:
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    unk_token = '<unk>'
    specials = [pad_token, sos_token, eos_token, unk_token]

    def __init__(self, tokenizer: Callable[[str], list[str]], vocab: Vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self._escape_pattern = re.compile(rf"({'|'.join(self.specials)})")
        self._unescape_pattern = re.compile(rf"#({'|'.join(self.specials)})")

    def _escape(self, text):
        return self._escape_pattern.sub(r'#\1', text)

    def _unescape(self, text):
        return self._unescape_pattern.sub(r'\1', text)

    def _tokenize(self, text: str) -> list[str]:
        return self.tokenizer(self._escape(text))

    def _detokenize(self, tokens: list[str]) -> str:
        return self._unescape(' '.join(tokens))

    def _index(self, tokens: list[str]) -> list[int]:
        return self.vocab.lookup_indices(tokens)

    def _deindex(self, indices: list[int]) -> list[str]:
        return self.vocab.lookup_tokens(indices)

    def encode(self, text: str) -> list[int]:
        return self._index(self._tokenize(text))

    def decode(self, indices: list[int]) -> str:
        return self._detokenize(self._deindex(indices))

    def __getitem__(self, token: str) -> int:
        return self.vocab[token]

    def __len__(self) -> int:
        return len(self.vocab)

    @staticmethod
    def load(path: Union[Path, str]) -> 'Tokenizer':
        tokenizer = torch.load(path, weights_only=False)
        assert isinstance(tokenizer, Tokenizer)
        return tokenizer


class TokenizerTrainer:
    def __init__(self, tokenizer: str = 'basic_english', language: str = 'en'):
        self.tokenizer = tokenizer
        self.language = language

    def train(self, texts: Iterable[str], num_tokens: int, save_path: Union[Path, str]) -> Tokenizer:
        tokenizer = get_tokenizer(self.tokenizer, self.language)
        batches = (tokenizer(text) for text in texts)
        vocab = build_vocab_from_iterator(batches, specials=Tokenizer.specials, max_tokens=num_tokens)
        vocab.set_default_index(vocab[Tokenizer.unk_token])
        tokenizer = Tokenizer(tokenizer, vocab)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokenizer, save_path)
        print(f'Saved tokenizer to: {save_path}')
        return tokenizer

    @staticmethod
    def get_pad_index() -> int:
        return Tokenizer.specials.index(Tokenizer.pad_token)
