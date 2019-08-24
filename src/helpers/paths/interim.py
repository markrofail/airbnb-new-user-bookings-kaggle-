from .. import paths

DATA_INTERIM_PATH = paths.DATA_PATH.joinpath('interim')


def train_dataset():
    return DATA_INTERIM_PATH.joinpath("train_users_2.csv")


def test_dataset():
    return DATA_INTERIM_PATH.joinpath("test_users.csv")
