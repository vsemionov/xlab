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

import re
from io import BytesIO
from pathlib import Path
from functools import partial
from typing import Union, Iterable
import warnings

import sentencepiece as spm


class Tokenizer:
    unk_token = '<unk>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    pad_token = '<pad>'
    specials = [unk_token, sos_token, eos_token, pad_token]

    _metaspace = '▁'  # U+2581
    _replacement = '<U-2581>'  # no regex special chars (use - instead of +)

    _escape_replacement = partial(re.compile(rf'(#*){_replacement}').sub, rf'\1\1#{_replacement}')  # 2k+1
    _replace_metaspace = partial(re.compile(rf'(#*){_metaspace}').sub, rf'\1\1{_replacement}')  # 2k
    _restore_metaspace = partial(re.compile(rf'(^|[^#])(#*)\2{_replacement}').sub, rf'\1\2{_metaspace}')  # 2k
    _unescape_replacement = partial(re.compile(rf'(^|[^#])(#*)\2#{_replacement}').sub, rf'\1\2{_replacement}')  # 2k+1

    def __init__(self, processor: spm.SentencePieceProcessor):
        self.processor = processor
        self._test()

    def _test(self):
        # make sure special token indices match expected order
        for index, token in enumerate(self.specials):
            assert self[token] == index

    @classmethod
    def _escape(cls, text):
        # regexes are slow, but contains checks are fast, so run replacements conditionally
        if isinstance(text, list):
            return [cls._escape(t) for t in text]
        if cls._replacement in text:
            text = cls._escape_replacement(text)
        if cls._metaspace in text:
            text = cls._replace_metaspace(text)
        return text

    @classmethod
    def _unescape(cls, text):
        if isinstance(text, list):
            return [cls._unescape(t) for t in text]
        if cls._replacement in text:
            text = cls._restore_metaspace(text)
            if cls._replacement in text:
                text = cls._unescape_replacement(text)
        return text

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(self._escape(text))

    def decode(self, indices: list[int]) -> str:
        return self._unescape(self.processor.decode(indices))

    def tokenize(self, text: str) -> list[str]:
        return self.processor.encode(self._escape(text), out_type=str)

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
    def load(path_or_data: Union[Path, str, bytes]) -> 'Tokenizer':
        processor = spm.SentencePieceProcessor()
        if isinstance(path_or_data, bytes):
            processor.load(model_proto=path_or_data)
        else:
            try:
                processor.load(str(path_or_data))
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

    def _chunk_line(self, line):
        # sentencepiece chokes on lines longer than 65536, so break them at spaces
        if len(line) <= 65536:
            yield line
            return
        chunks = line.split(' ')
        n = len(chunks)
        if n == 1:
            warnings.warn('Line too long and no more splits possible')
            return
        n = n // 2
        left = ' '.join(chunks[:n])
        right = ' '.join(chunks[n:])
        for chunk in self._chunk_line(left):
            yield chunk
        for chunk in self._chunk_line(right):
            yield chunk

    def train(self, texts: Iterable[str], num_tokens: int, save_path: Union[Path, str]) -> Tokenizer:
        save_path = Path(save_path)
        assert not save_path.exists()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        lines = (Tokenizer._escape(line) for text in texts for line in text.split('\n') if line)
        chunks = (chunk for line in lines for chunk in self._chunk_line(line) if chunk)
        model = BytesIO()
        kwargs = {
            **self.train_args,
            'input_format': 'text',
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
        spm.SentencePieceTrainer.train(sentence_iterator=chunks, model_writer=model, **kwargs)
        save_path.write_bytes(model.getvalue())
        print(f'Saved tokenizer to: {save_path}')
        return Tokenizer.load(model.getvalue())

    @staticmethod
    def get_pad_index() -> int:
        return Tokenizer.specials.index(Tokenizer.pad_token)


if __name__ == '__main__':
    def _test():
        import random
        import time

        path = Path('tokenizers/test.tok')
        if path.exists():
            path.unlink()

        tokenizer = TokenizerTrainer().train(['April is a month'], 299, path)
        path.unlink()

        text = 'Hello world!'
        assert tokenizer.decode(tokenizer.encode(text)) == text

        texts = ['Hello world!', 'Batch test']
        assert tokenizer.decode(tokenizer.encode(texts)) == texts

        text = '▁<U-2581>'
        assert tokenizer.decode(tokenizer.encode(text)) == text

        text = '#<U-2581>#▁'
        assert tokenizer.decode(tokenizer.encode(text)) == text

        random.seed(42)
        long_line = ''.join(chr(random.randint(32, 127)) for _ in range(65536 * 2))
        splits = list(TokenizerTrainer()._chunk_line(long_line))
        assert all(len(split) <= 65536 for split in splits)
        assert ' '.join(splits) == long_line

        sentence = 'The quick brown fox jumps over the lazy dog.'
        tokenizer = TokenizerTrainer().train([sentence], 358, path)
        path.unlink()
        reps = 10_000
        sentences = [sentence] * reps
        text = ' '.join([sentence] * reps)
        t0 = time.time()
        indices = tokenizer.encode(text)
        t1 = time.time()
        tokenizer.decode(indices)
        t2 = time.time()
        for _ in range(reps):
            indices = tokenizer.encode(sentence)
        t3 = time.time()
        for _ in range(reps):
            tokenizer.decode(indices)
        t4 = time.time()
        indices = tokenizer.encode(sentences)
        t5 = time.time()
        tokenizer.decode(indices)
        t6 = time.time()
        print(
            f'Encode speed: {len(text) / (t1 - t0) / 1024**2:,.1f} MB/s for long text,'
            f' {len(sentence) * reps / (t3 - t2) / 1024**2:,.1f} MB/s for short text'
            f' {len(sentence) * reps / (t5 - t4) / 1024**2:,.1f} MB/s for batched short text'
        )
        print(
            f'Decode speed: {len(text) / (t2 - t1) / 1024**2:,.1f} MB/s for long text,'
            f' {len(sentence) * reps / (t4 - t3) / 1024**2:,.1f} MB/s for short text'
            f' {len(sentence) * reps / (t6 - t5) / 1024**2:,.1f} MB/s for batched short text'
        )

        print('All OK')

    _test()
