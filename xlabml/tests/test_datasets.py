import unittest

import numpy as np
import torch

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

    def test_unconcatenated_unpadded_untrainedsos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=False, pad_incomplete=False,
            train_sos=False,
        )

        self.assertEqual(len(dataset), 3)

        x, y = dataset[0]
        self.assertEqual(x.tolist(), [1, 4, 4])
        self.assertEqual(y.tolist(), [4, 4, 2])

        x, y = dataset[1]
        self.assertEqual(x.tolist(), [1, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 5])

        x, y = dataset[2]
        self.assertEqual(x.tolist(), [5, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 2])

    def test_unconcatenated_unpadded_trainedsos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=False, pad_incomplete=False,
            train_sos=True,
        )

        self.assertEqual(len(dataset), 4)

        x, y = dataset[0]  # due to opportunistic sos
        self.assertEqual(x.tolist(), [1, 3, 2])
        self.assertEqual(y.tolist(), [3, 2, 1])

        x, y = dataset[1]
        self.assertEqual(x.tolist(), [1, 4, 4])
        self.assertEqual(y.tolist(), [4, 4, 2])

        x, y = dataset[2]
        self.assertEqual(x.tolist(), [1, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 5])

        x, y = dataset[3]
        self.assertEqual(x.tolist(), [5, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 2])

    def test_unconcatenated_padded_untrainedsos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=False, pad_incomplete=True,
            train_sos=False,
        )

        self.assertEqual(len(dataset), 6)

        x, y = dataset[0]
        self.assertEqual(x.tolist(), [1, 3, 2])
        self.assertEqual(y.tolist(), [3, 2, 0])

        x, y = dataset[1]
        self.assertEqual(x.tolist(), [1, 4, 4])
        self.assertEqual(y.tolist(), [4, 4, 2])

        x, y = dataset[2]
        self.assertEqual(x.tolist(), [4, 2, 0])
        self.assertEqual(y.tolist(), [2, 0, 0])

        x, y = dataset[3]
        self.assertEqual(x.tolist(), [1, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 5])

        x, y = dataset[4]
        self.assertEqual(x.tolist(), [5, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 2])

        x, y = dataset[5]
        self.assertEqual(x.tolist(), [5, 2, 0])
        self.assertEqual(y.tolist(), [2, 0, 0])

    def test_unconcatenated_padded_trainedsos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=False, pad_incomplete=True,
            train_sos=True,
        )

        self.assertEqual(len(dataset), 7)

        x, y = dataset[0]
        self.assertEqual(x.tolist(), [1, 3, 2])
        self.assertEqual(y.tolist(), [3, 2, 1])

        x, y = dataset[1]
        self.assertEqual(x.tolist(), [2, 1, 0])
        self.assertEqual(y.tolist(), [1, 0, 0])

        x, y = dataset[2]
        self.assertEqual(x.tolist(), [1, 4, 4])
        self.assertEqual(y.tolist(), [4, 4, 2])

        x, y = dataset[3]
        self.assertEqual(x.tolist(), [4, 2, 1])
        self.assertEqual(y.tolist(), [2, 1, 0])

        x, y = dataset[4]
        self.assertEqual(x.tolist(), [1, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 5])

        x, y = dataset[5]
        self.assertEqual(x.tolist(), [5, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 2])

        x, y = dataset[6]
        self.assertEqual(x.tolist(), [5, 2, 1])
        self.assertEqual(y.tolist(), [2, 1, 0])

    def test_concatenated_unpadded_untrainedsos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=False,
            train_sos=False,
        )

        self.assertEqual(len(dataset), 5)

        x, y, mask = dataset[0]
        self.assertEqual(x.tolist(), [1, 3, 2])
        self.assertEqual(y.tolist(), [3, 2, 0])

        x, y, mask = dataset[1]
        self.assertEqual(x.tolist(), [2, 1, 4])
        self.assertEqual(y.tolist(), [0, 4, 4])

        x, y, mask = dataset[2]
        self.assertEqual(x.tolist(), [4, 4, 2])
        self.assertEqual(y.tolist(), [4, 2, 0])

        x, y, mask = dataset[3]
        self.assertEqual(x.tolist(), [2, 1, 5])
        self.assertEqual(y.tolist(), [0, 5, 5])

        x, y, mask = dataset[4]
        self.assertEqual(x.tolist(), [5, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 5])

    def test_concatenated_unpadded_trainedsos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=False,
            train_sos=True,
        )

        self.assertEqual(len(dataset), 6)

        x, y, mask = dataset[0]
        self.assertEqual(x.tolist(), [1, 3, 2])
        self.assertEqual(y.tolist(), [3, 2, 1])

        x, y, mask = dataset[1]
        self.assertEqual(x.tolist(), [2, 1, 4])
        self.assertEqual(y.tolist(), [1, 4, 4])

        x, y, mask = dataset[2]
        self.assertEqual(x.tolist(), [4, 4, 2])
        self.assertEqual(y.tolist(), [4, 2, 1])

        x, y, mask = dataset[3]
        self.assertEqual(x.tolist(), [2, 1, 5])
        self.assertEqual(y.tolist(), [1, 5, 5])

        x, y, mask = dataset[4]
        self.assertEqual(x.tolist(), [5, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 5])

        x, y, mask = dataset[5]  # due to opportunistic sos
        self.assertEqual(x.tolist(), [5, 5, 2])
        self.assertEqual(y.tolist(), [5, 2, 1])

    def test_concatenated_padded_untrainedsos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=False,
        )

        self.assertEqual(len(dataset), 6)

        x, y, mask = dataset[0]
        self.assertEqual(x.tolist(), [1, 3, 2])
        self.assertEqual(y.tolist(), [3, 2, 0])

        x, y, mask = dataset[1]
        self.assertEqual(x.tolist(), [2, 1, 4])
        self.assertEqual(y.tolist(), [0, 4, 4])

        x, y, mask = dataset[2]
        self.assertEqual(x.tolist(), [4, 4, 2])
        self.assertEqual(y.tolist(), [4, 2, 0])

        x, y, mask = dataset[3]
        self.assertEqual(x.tolist(), [2, 1, 5])
        self.assertEqual(y.tolist(), [0, 5, 5])

        x, y, mask = dataset[4]
        self.assertEqual(x.tolist(), [5, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 5])

        x, y, mask = dataset[5]
        self.assertEqual(x.tolist(), [5, 5, 2])
        self.assertEqual(y.tolist(), [5, 2, 0])

    def test_concatenated_padded_trainedsos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        self.assertEqual(len(dataset), 7)

        x, y, mask = dataset[0]
        self.assertEqual(x.tolist(), [1, 3, 2])
        self.assertEqual(y.tolist(), [3, 2, 1])

        x, y, mask = dataset[1]
        self.assertEqual(x.tolist(), [2, 1, 4])
        self.assertEqual(y.tolist(), [1, 4, 4])

        x, y, mask = dataset[2]
        self.assertEqual(x.tolist(), [4, 4, 2])
        self.assertEqual(y.tolist(), [4, 2, 1])

        x, y, mask = dataset[3]
        self.assertEqual(x.tolist(), [2, 1, 5])
        self.assertEqual(y.tolist(), [1, 5, 5])

        x, y, mask = dataset[4]
        self.assertEqual(x.tolist(), [5, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 5])

        x, y, mask = dataset[5]
        self.assertEqual(x.tolist(), [5, 5, 2])
        self.assertEqual(y.tolist(), [5, 2, 1])

        x, y, mask = dataset[6]
        self.assertEqual(x.tolist(), [2, 1, 0])
        self.assertEqual(y.tolist(), [1, 0, 0])

    def test_mask_zero_sos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        x = torch.tensor([1, 3, 0, 4, 5, 6])
        mask = dataset._compute_mask(x)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ]
        self.assertEqual(mask.tolist(), expected)

    def test_mask_one_sos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        x = torch.tensor([3, 2, 1, 4, 5, 6])
        mask = dataset._compute_mask(x)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1],
        ]
        self.assertEqual(mask.tolist(), expected)

    def test_mask_one_sos_left(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        x = torch.tensor([1, 2, 3, 4, 5, 6])
        mask = dataset._compute_mask(x)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ]
        self.assertEqual(mask.tolist(), expected)

    def test_mask_one_sos_right(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        x = torch.tensor([0, 2, 3, 4, 5, 1])
        mask = dataset._compute_mask(x)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
        self.assertEqual(mask.tolist(), expected)

    def test_mask_two_sos(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        x = torch.tensor([2, 1, 2, 1, 5, 5])
        mask = dataset._compute_mask(x)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ]
        self.assertEqual(mask.tolist(), expected)

    def test_mask_two_sos_left(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        x = torch.tensor([1, 1, 2, 0, 5, 5])
        mask = dataset._compute_mask(x)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1],
        ]
        self.assertEqual(mask.tolist(), expected)

    def test_mask_two_sos_right(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        x = torch.tensor([2, 3, 2, 3, 1, 1])
        mask = dataset._compute_mask(x)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
        self.assertEqual(mask.tolist(), expected)

    def test_mask_two_sos_middle(self):
        dataset = SequenceDataset(
              parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            concatenate=True, pad_incomplete=True,
            train_sos=True,
        )

        x = torch.tensor([2, 3, 1, 1, 2, 3])
        mask = dataset._compute_mask(x)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ]
        self.assertEqual(mask.tolist(), expected)
