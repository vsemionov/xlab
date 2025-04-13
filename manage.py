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

import sys
from pathlib import Path
from io import BytesIO
import hashlib
import shutil

import torch
import click

from xlabml.utils import get_cache_dir


@click.group()
def manage():
    """Data and model management"""


@manage.command()
@click.argument('checkpoint_path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def export_checkpoint(checkpoint_path):
    """Export a clean checkpoint without training state"""
    checkpoint_keys = [
        'pytorch-lightning_version',
        'state_dict',
        'hyper_parameters',
        'datamodule_hyper_parameters',
    ]
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint = {k: checkpoint[k] for k in checkpoint_keys}
    with BytesIO() as output:
        torch.save(checkpoint, output)
        data = output.getvalue()
        hash = hashlib.sha256(data).hexdigest()
    output_path = checkpoint_path.parent / f'{checkpoint_path.stem}-{hash[:8]}.pt'
    if output_path.exists():
        print(f'Output file already exists: {output_path}', file=sys.stderr)
        exit(1)
    output_path.write_bytes(data)
    print(f'Saved to: {output_path}')


@manage.command()
def clear_cache():
    """Clear cache"""
    cache_dir = get_cache_dir(ensure_exists=False)
    if cache_dir.exists():
        print(f'Removing {cache_dir}')
        shutil.rmtree(cache_dir)


if __name__ == '__main__':
    manage()
