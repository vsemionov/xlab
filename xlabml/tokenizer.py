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

# ref: https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqazN3d29ySXRSbDRMc1diRXFlSmNOR0tYcTNVd3xBQ3Jtc0tsXzVpM2pEVS01SGFnQjRaTlBRdHZtbXpKNEN2ZER0UmR3T2xiLTRLSEFKVEk0QVNXOXRrWjdrZHRXNmZBaHVwWkJLZjFhemNlQXZJbm9tMTdUaFRHMC1TYTVtbmdjX0hINFdaY1FIbDRlOVJ5bXlqbw&q=https%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L%3Fusp%3Dsharing&v=zduSFxRajkE

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

    def get_token(self, index: int) -> str:
        return self.processor.id_to_piece(index)

    def __getitem__(self, token: str) -> int:
        return self.processor.piece_to_id(token)

    def __len__(self) -> int:
        return self.processor.get_piece_size()

    def is_learned(self, index):
        return not self.processor.is_unknown(index) \
            and not self.processor.is_control(index) \
            and not self.processor.is_byte(index)

    def is_char(self, index):
        return len(self.get_token(index)) == 1 and self.is_learned(index)

    @staticmethod
    def load(path: Union[Path, str]) -> 'Tokenizer':
        processor = spm.SentencePieceProcessor()
        try:
            processor.load(str(path))
        except OSError as e:
            raise FileNotFoundError from e
        return Tokenizer(processor)


class TokenizerTrainer:
    def __init__(self, train_args=None):
        # https://github.com/google/sentencepiece/blob/master/doc/options.md
        train_args = {
            'model_type': 'bpe',
            'character_coverage': 0.9995,
            'input_sentence_size': 0,  # max number of sentences
            'shuffle_input_sentence': True,
            'num_threads': 4,
            'max_sentencepiece_length': 16,
            'max_sentence_length': 1073741824,
            'split_by_unicode_script': True,
            'split_by_number': True,
            'split_by_whitespace': True,
            'split_digits': True,
            'treat_whitespace_as_suffix': False,
            'allow_whitespace_only_pieces': True,
            'required_chars': '',
            'byte_fallback': True,
            'normalization_rule_name': 'identity',
            'add_dummy_prefix': True,
            'remove_extra_whitespaces': False,
            **(train_args or {})
        }
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
            'hard_vocab_limit': True,
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
