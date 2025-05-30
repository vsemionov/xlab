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

import csv
import base64
from pathlib import Path
from typing import Any
import multiprocessing

import torch
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.cli import LightningCLI
from dotenv import load_dotenv

from xlabml.callbacks import *  # noqa
from xlabml.datamodules import XLabDataModule
from xlabml.models import XLabModule, XLabModel
from xlabml.sched import *
from xlabml.stats import compute_stats
from xlabml.utils import progress_bar
from xlabml import CONF_DIR


class XLabTrainer(Trainer):
    def download_data(self, model, datamodule: XLabDataModule):
        """Download source data"""
        datamodule.create_datasets_and_tokenizer(level='text')

    def encode_data(self, model, datamodule: XLabDataModule):
        """Encode text data"""
        datamodule.create_datasets_and_tokenizer(level='encode')

    def prepare_data(self, model, datamodule: XLabDataModule):
        """Prepare training data"""
        datamodule.prepare_data()

    def validate_data(self, model, datamodule: XLabDataModule, output_path: Path = 'invalid.csv', dump: bool = False):
        """Validate training data"""
        text_dataset = datamodule.create_datasets_and_tokenizer(['train'], level='text')['train']
        print(f'Writing results to: {output_path}')
        texts = progress_bar(text_dataset, kind=datamodule.progress, desc='Validating')
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['index', *(['text_b64'] if dump else [])]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            num_invalid = 0
            for idx, text in enumerate(texts):
                if not datamodule.tokenizer_trainer.validate_input(text):
                    num_invalid += 1
                    writer.writerow({
                        'index': idx,
                        **({'text_b64': base64.b64encode(text.encode()).decode()} if dump else {}),
                    })
        print(f'Results: {len(text_dataset)} total, {len(text_dataset) - num_invalid} valid, {num_invalid} invalid')

    def train_tokenizer(self, model, datamodule: XLabDataModule):
        """Train the tokenizer"""
        if datamodule.tokenizer_path.exists():
            print(f'Tokenizer already exists: {datamodule.tokenizer_path}')
            return
        datamodule.create_datasets_and_tokenizer(['train'], level='tokenizer', force_train=True)

    def compute_stats(self, model, datamodule: XLabDataModule, split: str = 'train', sample_size: int = 10_000):
        """Compute vocabulary and dataset statistics"""
        datamodule.prepare_data()
        tokenizer = datamodule.tokenizer
        dataset = datamodule.datasets[split]
        stats = compute_stats(tokenizer, dataset, sample_size=sample_size)
        print(
            f'Vocabulary: {len(tokenizer)} total,'
            f' {stats["vocabulary"]["num_learned"]} learned,'
            f' {stats["vocabulary"]["num_chars"]} characters,'
            f' {stats["vocabulary"]["num_unicode"]} unicode'
        )
        print(f'Split: {split}')
        print(
            f'Size: {len(dataset.parent):,} texts,'
            f' {stats["dataset"]["text_size_est"]:,} characters (est.),'
            f' {stats["dataset"]["token_size_est"]:,} tokens (est.),'
            f' {len(dataset):,} sequences'
        )
        print(
            f'Text length: {stats["dataset"]["text_length_mean"]:,.1f}'
            f' ({stats["dataset"]["text_length_median"]:,.1f})'
            f' ± {stats["dataset"]["text_length_std"]:,.1f} characters,'
            f' {stats["dataset"]["token_length_mean"]:,.1f}'
            f' ({stats["dataset"]["token_length_median"]:,.1f})'
            f' ± {stats["dataset"]["token_length_std"]:,.1f} tokens'
        )
        print(
            f'Mean token length in split: '
            f'{stats["dataset"]["text_size_est"] / stats["dataset"]["token_size_est"]:,.2f} characters'
        )
        print(f'Mean sequence fill ratio: {stats["dataset"]["seq_fill_ratio_mean"]:.2f}')


class XLabCLI(LightningCLI):
    data_subcommands = {
        'download_data',
        'encode_data',
        'prepare_data',
        'validate_data',
        'train_tokenizer',
        'compute_stats',
    }

    def __init__(self, *args: Any, trainer_class: type[XLabTrainer] = XLabTrainer, **kwargs: Any):
        super().__init__(*args, trainer_class=trainer_class, **kwargs)

    @classmethod
    def subcommands(cls):
        return {
            **super().subcommands(),
            **{subcommand: {'model', 'datamodule'} for subcommand in cls.data_subcommands},
        }

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('model.max_len', 'data.seq_len')
        parser.link_arguments('data.num_tokens', 'model.n_vocab')
        parser.link_arguments('data.tokenizer_trainer', 'model.pad_index',
            lambda tokenizer_trainer: tokenizer_trainer.get_pad_index(), apply_on='instantiate')

    def configure_optimizers(self, lightning_module, optimizer, lr_scheduler=None):
        if isinstance(lr_scheduler, XLabLRScheduler):
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    **lr_scheduler.config,
                },
            }
        else:
            return super().configure_optimizers(lightning_module, optimizer, lr_scheduler)


def main():
    load_dotenv()

    torch.set_float32_matmul_precision('high')
    multiprocessing.set_start_method('fork')  # needed on macos
    delattr(XLabModule, 'configure_optimizers')  # prevents a warning that method will be overridden by configuration

    parser_kwargs: dict = {
        subcommand: {
            'default_config_files': [
                CONF_DIR / 'defaults.yaml',
                *([CONF_DIR / 'extra/dummy.yaml'] if subcommand in XLabCLI.data_subcommands else []),
            ]
        }
        for subcommand in XLabCLI.subcommands()
    }
    parser_kwargs['parser_mode'] = 'omegaconf'

    XLabCLI(XLabModel, XLabDataModule, parser_kwargs=parser_kwargs)


if __name__ == '__main__':
    main()
