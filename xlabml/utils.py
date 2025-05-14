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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def progress_bar(iterable: Iterable, kind: str = 'tqdm', total: Optional[float] = None, desc: str = 'Working', **kwargs):
    if kind == 'tqdm':
        return tqdm(iterable, total=total, desc=desc, **kwargs)
    elif kind == 'rich':
        return track(iterable, total=total, description=desc, **kwargs)
    else:
        assert False


def requests_retry_session(
        session=None,  # existing session to decorate
        retries=3,  # number of total retries, excluding the initial attempt
        connect=None,  # number of retries on connect errors, 0 or False to disable, None to use the total retries
        read=None,  # number of retries on read errors, 0 or False to disable, None to use the total retries
        redirect=None,  # max number of redirects (see below), 0 or False to disable, None to use the total retries
        status=None,  # number of retries on bad status, 0 to disable, None to use the total retries
        other=None,  # number of retries on other errors (possibly after data sent), 0 to disable, None to use total
        allowed_methods=Retry.DEFAULT_ALLOWED_METHODS,  # limits read and status (but not connect/other) error retries
        status_forcelist=(429, 500, 502, 503, 504),  # unconditionally retry on these statuses, besides default ones
        backoff_factor=0.5  # first retry is immediate, second is after 2x this time
):
    # redirects are handled by the requests library and limited by Session.max_redirects
    # the retry counters are reset for each redirect
    # the Retry.redirect value has no effect

    session = session or requests.Session()

    session.max_redirects = retries if redirect is None else int(redirect)

    prefixes = ('http://', 'https://')

    for prefix, adapter in session.adapters.items():
        if prefix in prefixes:
            adapter.close()

    for prefix in prefixes:
        retry = Retry(
            total=retries,
            connect=connect,
            read=read,
            redirect=redirect,
            status=status,
            other=other,
            allowed_methods=allowed_methods,
            status_forcelist=status_forcelist,
            backoff_factor=backoff_factor,
            respect_retry_after_header=False  # prevent server-controlled delays
        )

        adapter = HTTPAdapter(max_retries=retry)

        session.mount(prefix, adapter)

    return session


def download(url: str, save_path: Union[Path, str], desc: Optional[str] = None):
    block_size = 32 * 1024
    desc = desc or save_path.name
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with requests_retry_session(retries=3) as session, \
            session.get(url, timeout=30, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(save_path, 'wb') as file, \
                tqdm(desc=desc, total=total, unit='B', unit_scale=True) as pbar:
            for data in response.iter_content(block_size):
                size = file.write(data)
                pbar.update(size)
