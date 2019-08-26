from .. import paths


MODELS_PATH = paths.ROOT_PATH.joinpath('models')


def _2048_1024_512(file=False):
    path = MODELS_PATH.joinpath('2048_1024_512')
    if file:
        path = path.joinpath("saved_model.pb")
    return path
