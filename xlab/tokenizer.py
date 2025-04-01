import re
from pathlib import Path
from typing import Optional, Union, Iterable

import torch
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

    def has_vocab(self) -> bool:
        return self.vocab is not None

    def build_vocab(self, batches: Iterable[list[str]]) -> 'Tokenizer':
        self.vocab = build_vocab_from_iterator(batches, specials=self.specials, max_tokens=self.max_tokens)
        self.vocab.set_default_index(self.vocab[self.unk_token])
        return self

    def load_vocab(self, path: Union[str, Path]) -> 'Tokenizer':
        self.vocab = torch.load(path)
        return self

    def save_vocab(self, path: Union[str, Path]) -> 'Tokenizer':
        torch.save(self.vocab, path)
        return self

    def reset_vocab(self) -> 'Tokenizer':
        self.vocab = None
        return self

    def vocab_size(self) -> int:
        return len(self.vocab)
