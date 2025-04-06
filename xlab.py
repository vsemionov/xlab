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
from lightning.pytorch.callbacks import TQDMProgressBar, RichProgressBar, RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme

from xlab.config import THEME_COLOR
from xlab.datamodules import XLabDataModule
from xlab.models import XLabModule, XLabModel


class XLabCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.max_tokens', 'model.n_vocab')
        parser.link_arguments('model.max_len', 'data.seq_len')
        parser.link_arguments('data.tokenizer', 'model.pad_index',
            lambda tokenizer: tokenizer.specials.index(tokenizer.pad_token), apply_on='instantiate')


class ProgressMixin:
    def __init__(self, abbreviate=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.abbreviate = abbreviate

    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)
        items.pop('v_num', None)
        if self.abbreviate:
            for metric in list(items):
                abbrev = ''.join(word[0] for word in metric.split('_'))
                assert abbrev not in items
                items[abbrev] = items.pop(metric)
        return items


class XLabTQDMProgressBar(ProgressMixin, TQDMProgressBar):
    def __init__(self, abbreviate: bool = True, refresh_rate: int = 1, leave: bool = False):
        super().__init__(abbreviate=abbreviate, refresh_rate=refresh_rate, leave=leave)


class XLabRichProgressBar(ProgressMixin, RichProgressBar):
    def __init__(self, abbreviate: bool = False, refresh_rate: int = 1, leave: bool = False):
        # workaround for failing config validation
        theme = RichProgressBarTheme(
            progress_bar=THEME_COLOR,
            progress_bar_finished=THEME_COLOR,
            progress_bar_pulse=THEME_COLOR
        )
        super().__init__(abbreviate=abbreviate, refresh_rate=refresh_rate, leave=leave, theme=theme)


class XLabRichModelSummary(RichModelSummary):
    def __init__(self, max_depth: int = 1):
        super().__init__(max_depth=max_depth, header_style='bold')


def main():
    torch.set_float32_matmul_precision('high')
    multiprocessing.set_start_method('fork')  # needed on macos
    delattr(XLabModule, 'configure_optimizers')  # prevents a warning that method will be overridden by configuration

    parser_kwargs = {stage: {'default_config_files': [Path(__file__).parent / 'conf' / f'defaults.yaml']}
        for stage in ['fit', 'validate', 'test', 'predict']}

    XLabCLI(XLabModel, XLabDataModule, parser_kwargs=parser_kwargs)


if __name__ == '__main__':
    main()
