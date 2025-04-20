import unittest

import numpy as np
import torch
import datasets as hf_datasets

from ..datasets import BaseDataset, SequenceDataset


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


class MockTokenDataset(BaseDataset):
    def __init__(self, arrays):
        column = 'indices'
        dataset = hf_datasets.Dataset.from_dict({column: arrays})
        super().__init__(column=column, parent=dataset, dataset=dataset)
        self.tokenizer = MockTokenizer()


class TestSequenceDataset(unittest.TestCase):
    def setUp(self):
        self.sample_data = [[3], [4, 4], [5, 5, 5, 5]]
        self.sample_data = [np.array(indices) for indices in self.sample_data]
        self.token_dataset = MockTokenDataset(self.sample_data)

    def test_basic(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
        )

        self.assertEqual(len(dataset), 6)

        x, y, m = dataset[0]
        self.assertEqual(x.tolist(), [1, 3, 2])
        self.assertEqual(y.tolist(), [3, 2, 0])
        self.assertEqual(
            m.tolist(),
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 0, 0]
            ]
        )

        x, y, m = dataset[1]
        self.assertEqual(x.tolist(), [1, 4, 4])
        self.assertEqual(y.tolist(), [4, 4, 2])
        self.assertEqual(
            m.tolist(),
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1]
            ]
        )

        x, y, m = dataset[2]
        self.assertEqual(x.tolist(), [4, 2, 0])
        self.assertEqual(y.tolist(), [2, 0, 0])
        self.assertEqual(
            m.tolist(),
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        )

        x, y, m = dataset[3]
        self.assertEqual(x.tolist(), [1, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 5])
        self.assertEqual(
            m.tolist(),
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1]
            ]
        )

        x, y, m = dataset[4]
        self.assertEqual(x.tolist(), [5, 5, 5])
        self.assertEqual(y.tolist(), [5, 5, 2])
        self.assertEqual(
            m.tolist(),
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1]
            ]
        )

        x, y, m = dataset[5]
        self.assertEqual(x.tolist(), [5, 2, 0])
        self.assertEqual(y.tolist(), [2, 0, 0])
        self.assertEqual(
            m.tolist(),
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        )

    def test_unpadded(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
            pad_incomplete=False,
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

    def test_batch_get(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
        )

        xyms = [dataset[i] for i in range(len(dataset))]
        batch_xyms = dataset[list(range(len(dataset)))]
        self.assertEqual(len(batch_xyms), len(xyms))

        for batch_xym, xym in zip(batch_xyms, xyms):
            x, y, m = xym
            batch_x, batch_y, batch_m = batch_xym
            self.assertEqual(x.tolist(), batch_x.tolist())
            self.assertEqual(y.tolist(), batch_y.tolist())
            self.assertEqual(m.tolist(), batch_m.tolist())

    def test_iter(self):
        dataset = SequenceDataset(
            parent=self.token_dataset,  # noqa
            seq_len=3, step_size=2,
        )

        xyms = [dataset[i] for i in range(len(dataset))]
        iter_xyms = list(dataset)
        self.assertEqual(len(iter_xyms), len(xyms))

        for iter_xym, xym in zip(iter_xyms, xyms):
            x, y, m = xym
            iter_x, iter_y, iter_m = iter_xym
            self.assertEqual(x.tolist(), iter_x.tolist())
            self.assertEqual(y.tolist(), iter_y.tolist())
            self.assertEqual(m.tolist(), iter_m.tolist())
