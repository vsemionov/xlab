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

from pathlib import Path
from typing import Union, Iterable

import sentencepiece as spm

class Tokenizer:
    unk_token = '<unk>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    pad_token = '<pad>'
    specials = [unk_token, sos_token, eos_token, pad_token]

    def __init__(self, processor: spm.SentencePieceProcessor):
        self.processor = processor
        for index, token in enumerate(self.specials):
            assert self[token] == index

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text)

    def decode(self, indices: list[int]) -> str:
        return self.processor.decode(indices)

    def __getitem__(self, token: str) -> int:
        return self.processor.piece_to_id(token)

    def __len__(self) -> int:
        return self.processor.get_piece_size()

    @staticmethod
    def load(path: Union[Path, str]) -> 'Tokenizer':
        processor = spm.SentencePieceProcessor()
        try:
            processor.load(str(path))
        except OSError as e:
            raise FileNotFoundError from e
        return Tokenizer(processor)


class TokenizerTrainer:
    def __init__(
            self,
            train_args: dict = {
            }
    ):
        self.train_args = train_args

    def train(self, texts: Iterable[str], num_tokens: int, save_path: Union[Path, str]) -> Tokenizer:
        save_path = Path(save_path)
        assert save_path.name.endswith('.model')
        assert not save_path.exists()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sentences = (line for text in texts for line in text.split('\n') if line)
        kwargs = {
            **self.train_args,
            'input_format': 'text',
            'model_prefix': save_path.parent / save_path.stem,
            'vocab_size': num_tokens,
            'unk_id': Tokenizer.specials.index(Tokenizer.unk_token),
            'bos_id': Tokenizer.specials.index(Tokenizer.sos_token),
            'eos_id': Tokenizer.specials.index(Tokenizer.eos_token),
            'pad_id': Tokenizer.specials.index(Tokenizer.pad_token),
            'unk_piece': Tokenizer.unk_token,
            'bos_piece': Tokenizer.sos_token,
            'eos_piece': Tokenizer.eos_token,
            'pad_piece': Tokenizer.pad_token,
        }
        spm.SentencePieceTrainer.train(sentence_iterator=sentences, **kwargs)
        return Tokenizer.load(save_path)

    @staticmethod
    def get_pad_index() -> int:
        return Tokenizer.specials.index(Tokenizer.pad_token)
