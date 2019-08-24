from .. import paths

DATA_RAW_PATH = paths.DATA_PATH.joinpath('raw')


def train_dataset():
    return DATA_RAW_PATH.joinpath("train_users_2.csv")


def test_dataset():
    return DATA_RAW_PATH.joinpath("test_users.csv")
