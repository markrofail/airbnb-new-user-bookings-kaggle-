from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.helpers import paths


def one():
    df = pd.read_csv("file.csv")

    n_rows = df.shape[0]
    columns = list(df['Unnamed: 0'])

    n = n_rows // 4
    cols = ['country_destination_AU',
            'country_destination_CA',
            'country_destination_DE',
            'country_destination_ES',
            'country_destination_FR',
            'country_destination_GB',
            'country_destination_IT',
            'country_destination_NDF',
            'country_destination_NL',
            'country_destination_PT',
            'country_destination_US',
            'country_destination_other']
    df = df[cols]
    df = df.abs()

    to_remove = list()
    for col in cols:
        to_remove.append([columns[i] for i in reversed(np.argsort(df[col])[:n])])

    cnt = Counter()
    for x in to_remove:
        for y in x:
            cnt[y] += 1

    to_remove = list(cnt.most_common(n))
    to_remove, _ = zip(*to_remove)
    to_remove = list(to_remove)
    to_remove.remove('country_destination_other')
    return to_remove


def apply():
    df = pd.read_csv(paths.processed.test_dataset())

    clm_to_remove = one()
    df.drop(columns=clm_to_remove, inplace=True)

    df.to_csv(paths.processed.test_dataset(), index=False)


if __name__ == "__main__":
    # one()
    apply()
