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

import torch
import platformdirs
from tqdm.auto import tqdm
from rich.progress import track

from . import config


def progress_bar(iterable, kind='tqdm', total=None, desc='Working'):
    if kind == 'tqdm':
        return tqdm(iterable, total=total, desc=desc)
    elif kind == 'rich':
        return track(iterable, total=total, description=desc)
    else:
        assert False


def get_cache_dir(ensure_exists=True):
    return platformdirs.user_cache_path(config.APP_NAME, ensure_exists=ensure_exists)


def cached(function, name, fingerprint):
    cache_dir = get_cache_dir()
    path = cache_dir / name / f'{fingerprint}.pt'
    try:
        result = torch.load(path)
    except FileNotFoundError:
        path.parent.mkdir(parents=True, exist_ok=True)
        result = function()
        torch.save(result, path)
    return result
