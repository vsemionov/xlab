import unittest
import sys

import numpy as np

from ..datasets import SequenceDataset


class MockTokenizer:
    def __init__(self):
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'

    def __getitem__(self, token):
        token_map = {
            '<sos>': 1,
            '<eos>': 2,
            '<pad>': 0,
        }
        return token_map[token]


class MockTextDataset:
    split = 'train'


class MockTokenDataset:
    tokenizer = MockTokenizer()
    parent = MockTextDataset()

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class TestSequenceDataset(unittest.TestCase):
    def setUp(self):
        self.sample_data = [[3], [4, 4], [5, 5, 5, 5]]
        self.sample_data = [np.array(indices) for indices in self.sample_data]
        self.token_dataset = MockTokenDataset(self.sample_data)

    def test_unconcatenated_unpadded_untrainedeos(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,
            seq_len=3, step_size=2,
            concatenate=False, pad_incomplete=False,
            train_sos=False,
        )

        self.assertEqual(len(dataset), 3)

        x, y = dataset[0]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 4, 4])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([4, 4, 2])).all())

        x, y = dataset[1]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 5])).all())

        x, y = dataset[2]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 2])).all())

    def test_unconcatenated_unpadded_trainedeos(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,
            seq_len=3, step_size=2,
            concatenate=False, pad_incomplete=False,
            train_sos=True,
        )

        self.assertEqual(len(dataset), 3)

        x, y = dataset[0]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 4, 4])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([4, 4, 2])).all())

        x, y = dataset[1]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 5])).all())

        x, y = dataset[2]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 2])).all())

    def test_unconcatenated_padded_untrainedeos(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,
            seq_len=3, step_size=2,
            concatenate=False, pad_incomplete=True,
            train_sos=False,
        )

        self.assertEqual(len(dataset), 6)

        x, y = dataset[0]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 3, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([3, 2, 0])).all())

        x, y = dataset[1]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 4, 4])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([4, 4, 2])).all())

        x, y = dataset[2]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([4, 2, 0])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([2, 0, 0])).all())

        x, y = dataset[3]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 5])).all())

        x, y = dataset[4]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 2])).all())

        x, y = dataset[5]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 2, 0])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([2, 0, 0])).all())

    def test_unconcatenated_padded_trainedeos(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,
            seq_len=3, step_size=2,
            concatenate=False, pad_incomplete=True,
            train_sos=True,
        )

        self.assertEqual(len(dataset), 7)

        x, y = dataset[0]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 3, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([3, 2, 1])).all())

        x, y = dataset[1]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 0])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([1, 0, 0])).all())

        x, y = dataset[2]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 4, 4])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([4, 4, 2])).all())

        x, y = dataset[3]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([4, 2, 1])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([2, 1, 0])).all())

        x, y = dataset[4]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 5])).all())

        x, y = dataset[5]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 2])).all())

        x, y = dataset[6]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 2, 1])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([2, 1, 0])).all())

    def test_concatenated_unpadded_untrainedeos(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=False,
            train_sos=False,
        )

        self.assertEqual(len(dataset), 5)

        x, y = dataset[0]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 3, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([3, 2, 0])).all())

        x, y = dataset[1]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 4])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([0, 4, 4])).all())

        x, y = dataset[2]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([4, 4, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([4, 2, 0])).all())

        x, y = dataset[3]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([0, 5, 5])).all())

        x, y = dataset[4]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 5])).all())

    def test_concatenated_unpadded_trainedeos(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=False,
            train_sos=True,
        )

        self.assertEqual(len(dataset), 5)

        x, y = dataset[0]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 3, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([3, 2, 1])).all())

        x, y = dataset[1]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 4])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([1, 4, 4])).all())

        x, y = dataset[2]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([4, 4, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([4, 2, 1])).all())

        x, y = dataset[3]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([1, 5, 5])).all())

        x, y = dataset[4]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 5])).all())

    def test_concatenated_padded_untrainedeos(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=False,
        )

        self.assertEqual(len(dataset), 6)

        x, y = dataset[0]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 3, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([3, 2, 0])).all())

        x, y = dataset[1]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 4])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([0, 4, 4])).all())

        x, y = dataset[2]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([4, 4, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([4, 2, 0])).all())

        x, y = dataset[3]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([0, 5, 5])).all())

        x, y = dataset[4]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 5])).all())

        x, y = dataset[5]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 2, 0])).all())

    def test_concatenated_padded_trainedeos(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        self.assertEqual(len(dataset), 7)

        x, y = dataset[0]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([1, 3, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([3, 2, 1])).all())

        x, y = dataset[1]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 4])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([1, 4, 4])).all())

        x, y = dataset[2]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([4, 4, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([4, 2, 1])).all())

        x, y = dataset[3]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([1, 5, 5])).all())

        x, y = dataset[4]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 5])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 5, 5])).all())

        x, y = dataset[5]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([5, 5, 2])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([5, 2, 1])).all())

        x, y = dataset[6]
        print(x, file=sys.stderr); self.assertTrue((x == np.array([2, 1, 0])).all())
        print(y, file=sys.stderr); self.assertTrue((y == np.array([1, 0, 0])).all())
