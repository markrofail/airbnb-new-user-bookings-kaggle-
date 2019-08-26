import yaml

from .. import paths


def config():
    return paths.ROOT_PATH.joinpath('config.yml')


def read(path=config()):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.BaseLoader)
