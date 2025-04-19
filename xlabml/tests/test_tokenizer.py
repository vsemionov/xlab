import unittest
import time
import random

from ..tokenizer import TokenizerTrainer
from .. import ROOT_DIR

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer_path = ROOT_DIR / 'tokenizers/test.tok'
        self.tokenizer = TokenizerTrainer().train(['April is a month'], 299, self.tokenizer_path)

    def tearDown(self):
        if self.tokenizer_path.exists():
            self.tokenizer_path.unlink()

    def test_encode_decode(self):
        text = 'Hello world!'
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_encode_decode_batch(self):
        texts = ['Hello world!', 'Batch test']
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(texts)), texts)

    def test_replacement(self):
        text = '▁<U-2581>'
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_escaped_replacement(self):
        text = '#<U-2581>#▁'
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_speed(self):
        sentence = 'The quick brown fox jumps over the lazy dog.'
        self.tokenizer_path.unlink()
        tokenizer = TokenizerTrainer().train([sentence], 358, self.tokenizer_path)
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



class TestTokenizerTrainer(unittest.TestCase):
    def test(self):
        random.seed(42)
        long_line = ''.join(chr(random.randint(32, 127)) for _ in range(65536 * 2))
        splits = list(TokenizerTrainer()._chunk_line(long_line))
        self.assertTrue(all(len(split) <= 65536 for split in splits))
        self.assertEqual(' '.join(splits), long_line)
