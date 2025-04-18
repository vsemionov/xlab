from pathlib import Path

from xlabml.tokenizer import TokenizerTrainer


if __name__ == '__main__':
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
