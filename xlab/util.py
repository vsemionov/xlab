import platformdirs
from tqdm.auto import tqdm
from rich.progress import track

from . import config


def progress_bar(iterable, kind='tqdm', desc='Working'):
    if kind == 'tqdm':
        return tqdm(iterable, desc=desc)
    elif kind == 'rich':
        return track(iterable, description=desc)
    else:
        assert False


def get_cache_dir():
    return platformdirs.user_cache_path(config.APP_NAME, ensure_exists=True)
