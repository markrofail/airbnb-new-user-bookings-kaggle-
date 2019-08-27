
import numpy as np
import pandas as pd

from src.data import utils


def fix_age(df):
    # Correct Formating
    # age > 1000 replace with year created - value
    df['age'] = np.where(df['age'] < 1000, df['age'],
                         df['date_account_created'].dt.year - df['age'])

    # Fill Missing Data
    df['age'].fillna(-1, inplace=True)
    df['age'] = df['age'].astype(int)


def age_categorization(df):
    bins = list(range(0, 110+1, 5))

    labels = ""
    start = True
    for interval in bins:
        if start:
            labels += str(interval)
            start = False
        else:
            labels += "-{},{}".format(interval-1, interval)
    labels += "+"
    labels = labels.split(',')
    labels = labels[:-1]
    df['ageGroup'] = pd.cut(df['age'], bins=bins, labels=labels)
    df.ageGroup.cat.add_categories(-1, inplace=True)
    df.ageGroup.fillna(-1, inplace=True)
    df.drop(columns=['age'], inplace=True)
    df = utils.onehot.one_hot_encode_clm(df, 'ageGroup')
    return df
