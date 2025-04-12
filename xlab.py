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
import multiprocessing

import torch
from lightning.pytorch.cli import LightningCLI

from xlabml.datamodules import XLabDataModule
from xlabml.models import XLabModule, XLabModel


class XLabCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.max_tokens', 'model.n_vocab')
        parser.link_arguments('model.max_len', 'data.seq_len')
        parser.link_arguments('data.tokenizer', 'model.pad_index',
            lambda tokenizer: tokenizer.specials.index(tokenizer.pad_token), apply_on='instantiate')


def main():
    torch.set_float32_matmul_precision('high')
    multiprocessing.set_start_method('fork')  # needed on macos
    delattr(XLabModule, 'configure_optimizers')  # prevents a warning that method will be overridden by configuration

    parser_kwargs = {stage: {'default_config_files': [Path(__file__).parent / 'conf' / f'defaults.yaml']}
        for stage in ['fit', 'validate', 'test', 'predict']}

    XLabCLI(XLabModel, XLabDataModule, parser_kwargs=parser_kwargs)


if __name__ == '__main__':
    main()
