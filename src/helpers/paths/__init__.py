from pathlib import Path


def _find_ROOT(abs_path):
    while abs_path.name != 'src':
        abs_path = abs_path.parent
    return abs_path.parent


def delete_folder(dir):
    for sub in dir.iterdir():
        if sub.is_dir():
            delete_folder(sub)
        else:
            sub.unlink()
    dir.rmdir()  # if you just want to delete dir content, remove this line


ROOT_PATH = _find_ROOT(Path(__file__).absolute())
DATA_PATH = ROOT_PATH.joinpath('data')

DATA_EXTERNAL_PATH = DATA_PATH.joinpath('external')


from . import (raw, interim, processed, models, config, results)  # noqa

__all__ = [
    'raw',
    'interim',
    'processed',
    'models'
    'config'
    'results'
]
