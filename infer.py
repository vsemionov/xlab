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
import warnings

import torch
import click

from xlabml.tokenizer import Tokenizer
from xlabml.datamodules import XLabDataModule
from xlabml.models import XLabModel
from xlabml import inference
from xlabml import text


@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('prompt')
@click.option('-t', '--tokenizer', 'tokenizer_path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('-c', '--continued', is_flag=True)
@click.option('-d', '--device', type=click.Choice(['auto', 'cuda', 'cpu']), default='auto')
@click.option('-n', '--runs', type=click.IntRange(min=1), default=1)
@click.option('-l', '--limit', type=click.IntRange(min=1), default=100)
@click.option('-T', '--temperature', type=click.FloatRange(min=0, min_open=True), default=1)
@click.option('-k', '--top-k', type=click.IntRange(min=1))
@click.option('-p', '--top-p', type=click.FloatRange(min=0, max=1, min_open=True))
@click.option('-s', '--seed', type=int)
@click.option('-b', '--beam-search', is_flag=True)
@click.option('-w', '--beam-width', type=click.IntRange(min=1), default=10)
@click.option('-z', '--length-penalty', type=float, default=0)
@click.option('--debug', is_flag=True)
def main(
        checkpoint_path, prompt, tokenizer_path, continued, device,
        runs, limit, temperature, top_k, top_p, seed,
        beam_search, beam_width, length_penalty,
        debug
):
    """Model inference"""

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if tokenizer_path is None:
        datamodule = XLabDataModule.load_from_checkpoint(checkpoint_path, map_location=device)
        tokenizer_path = datamodule.tokenizer_path
    tokenizer = Tokenizer.load(tokenizer_path)

    with warnings.catch_warnings():
        if not debug:
            warnings.simplefilter('ignore')  # prevents warning about unnecessary hyperparameter 'activation'
        model = XLabModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval().requires_grad_(False)

    if len(tokenizer) != model.hparams['n_vocab']:
        raise ValueError(f'Tokenizer vocabulary size {len(tokenizer)} differs from model {model.hparams["n_vocab"]}')

    sos_index, eos_index = [tokenizer[token] for token in [tokenizer.sos_token, tokenizer.eos_token]]
    prefix = [] if continued else [sos_index]
    inputs = torch.tensor(prefix + tokenizer.encode(prompt), device=model.device)
    max_len = model.hparams['max_len']
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    for _ in range(runs):
        if not beam_search:
            indices = inference.sample(
                model, inputs,
                temperature=temperature, top_k=top_k, top_p=top_p, generator=generator,
                output_length=limit, block_size=max_len, eos_class=eos_index, exclude_classes=None,
            )
        else:
            indices = inference.beam_search(
                model, inputs,
                beam_width=beam_width, length_penalty=length_penalty,
                output_length=limit, block_size=max_len, eos_class=eos_index, exclude_classes=None,
            )

        rev_prompt = tokenizer.decode(inputs[len(prefix):].tolist())
        indices = indices.tolist()
        if len(indices) < limit:
            indices.append(eos_index)
        output = tokenizer.decode(indices)
        sep = ' ' if rev_prompt else ''
        escaped = text.escape(f'{rev_prompt}{sep}{output}')
        print(escaped)


if __name__ == '__main__':
    main()
