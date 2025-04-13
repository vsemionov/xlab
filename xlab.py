#!/usr/bin/env python

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
from typing import Any
import multiprocessing

import torch
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.cli import LightningCLI

from xlabml.callbacks import *  # noqa
from xlabml.datamodules import XLabDataModule
from xlabml.models import XLabModule, XLabModel
from xlabml.stats import compute_stats


class XLabTrainer(Trainer):
    def train_tokenizer(self, model, datamodule: XLabDataModule):
        """Train the tokenizer"""
        if datamodule.tokenizer_path.exists():
            print(f'Tokenizer already exists: {datamodule.tokenizer_path}')
            return
        datamodule.create_datasets_and_tokenizer(['train'], tokenizer_only=True)

    def compute_stats(self, model, datamodule: XLabDataModule, split: str = 'train', sample_size: int = 10_000):
        """Compute dataset statistics"""
        datamodule.prepare_data()
        dataset = datamodule.datasets[split].dataset
        stats = compute_stats(dataset, sample_size=sample_size)
        print(f'Split: {split}')
        print(f'Size: {len(dataset):,} texts, {stats["size_est"]:,} tokens (est.)')
        print(
            f'Text length: {stats["length_mean"]:,.1f}'
            f' ({stats["length_median"]:,.1f})'
            f' ± {stats["length_std"]:,.1f} tokens'
        )


class XLabCLI(LightningCLI):
    data_subcommands = {'compute_stats'}

    def __init__(self, *args: Any, trainer_class: type[XLabTrainer] = XLabTrainer, **kwargs: Any):
        super().__init__(*args, trainer_class=trainer_class, **kwargs)

    @classmethod
    def subcommands(cls):
        return {
            **super().subcommands(),
            'compute_stats': {'model', 'datamodule'},
            'train_tokenizer': {'model', 'datamodule'},
        }

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('model.max_len', 'data.seq_len')
        parser.link_arguments('data.num_tokens', 'model.n_vocab')
        parser.link_arguments('data.tokenizer_trainer', 'model.pad_index',
            lambda tokenizer_trainer: tokenizer_trainer.get_pad_index(), apply_on='instantiate')


def main():
    torch.set_float32_matmul_precision('high')
    multiprocessing.set_start_method('fork')  # needed on macos
    delattr(XLabModule, 'configure_optimizers')  # prevents a warning that method will be overridden by configuration

    conf_dir = Path(__file__).parent / 'conf'
    parser_kwargs = {
        subcommand: {
            'default_config_files': [
                conf_dir / 'defaults.yaml',
                *([conf_dir / 'extra/dummy.yaml'] if subcommand in XLabCLI.data_subcommands else []),
            ]
        }
        for subcommand in XLabCLI.subcommands()
    }

    XLabCLI(XLabModel, XLabDataModule, parser_kwargs=parser_kwargs)


if __name__ == '__main__':
    main()
