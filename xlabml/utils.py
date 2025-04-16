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
from typing import Optional, Union, Iterable

import requests
from tqdm.auto import tqdm
from rich.progress import track


def progress_bar(iterable: Iterable, kind: str = 'tqdm', total: Optional[float] = None, desc: str = 'Working'):
    if kind == 'tqdm':
        return tqdm(iterable, total=total, desc=desc)
    elif kind == 'rich':
        return track(iterable, total=total, description=desc)
    else:
        assert False


def download(url: str, save_path: Union[Path, str], desc: Optional[str] = None):
    block_size = 32 * 1024
    desc = desc or save_path.name
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with (requests.get(url, stream=True) as response):
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, \
                tqdm(desc=desc, total=total, unit='B', unit_scale=True) as pbar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                pbar.update(size)
