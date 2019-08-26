from .. import paths

DATA_PROCESSED_PATH = paths.DATA_PATH.joinpath('processed')


def train_dataset():
    return DATA_PROCESSED_PATH.joinpath("train_features.csv")


def test_dataset():
    return DATA_PROCESSED_PATH.joinpath("test_features.csv")


def train_labels():
    return DATA_PROCESSED_PATH.joinpath("train_labels.csv")


def test_labels():
    return DATA_PROCESSED_PATH.joinpath("test_labels.csv")
