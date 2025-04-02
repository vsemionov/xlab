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

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import TQDMProgressBar, RichProgressBar
from lightning.pytorch.loggers import Logger

from xlab.config import APP_NAME
from xlab.dataset import XLabDataModule
from xlab.models import XLabModel


class XLabCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data.max_tokens', 'model.n_vocab')
        parser.link_arguments('model.max_len', 'data.seq_len')
        parser.link_arguments('data.tokenizer', 'model.pad_index',
            lambda tokenizer: tokenizer.specials.index(tokenizer.pad_token), apply_on='instantiate')


class Progress:
    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)
        items.pop('v_num', None)
        return items


class XLabTQDMProgressBar(Progress, TQDMProgressBar):
    pass


class XLabRichProgressBar(Progress, RichProgressBar):
    def __init__(self):
        # workaround for failing config validation
        super().__init__()


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage):
        super().save_config(trainer, pl_module, stage)
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams({'config': config})


def main():
    multiprocessing.set_start_method('fork')  # needed on macos

    parser_kwargs = {stage: {'default_config_files': [Path(__file__).parent / 'conf' / f'{APP_NAME}.yaml']}
        for stage in ['fit', 'validate', 'test', 'predict']}

    XLabCLI(XLabModel, XLabDataModule, parser_kwargs=parser_kwargs, save_config_callback=LoggerSaveConfigCallback)

if __name__ == '__main__':
    main()
