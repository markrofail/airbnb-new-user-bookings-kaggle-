from .. import paths


RESULTS_PATH = paths.ROOT_PATH.joinpath('results')


def _2048_1024_512():
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    path = RESULTS_PATH.joinpath('2048_1024_512.csv')
    return path
