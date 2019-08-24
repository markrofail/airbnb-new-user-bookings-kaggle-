import numpy as np
import pandas as pd

from src.helpers import paths

import matplotlib.pyplot as plt

def fix_date(df):
    # Correct Formating
    df['date_account_created'] = pd.to_datetime(df['date_account_created'])
    df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'])
    df['date_first_booking'] = pd.to_datetime(df['date_first_booking'])

    # Remove Non essential data
    df.drop('date_first_booking', axis=1, inplace=True)


def fix_age(df):
    # Correct Formating
    # age > 1000 replace with year created - value
    df['age'] = np.where(df['age'] < 1000, df['age'],
                         df['date_account_created'].dt.year - df['age'])

    # Fill Missing Data
    df['age'].fillna(-1, inplace=True)
    df['age'] = df['age'].astype(int)


def fix_affliate(df):
    # Fill Missing Data
    df['first_affiliate_tracked'].fillna("-unknown-", inplace=True)


def compute_corr(df):
    pass
    # corr = df.corr()
    # corr.style.background_gradient(cmap='coolwarm', axis=None)
    # plt.matshow(corr)
    # plt.show()


def clean_data(df):
    fix_date(df)
    fix_age(df)
    fix_affliate(df)


def apply():
    train_path = paths.raw.train_dataset()
    train_data = pd.read_csv(train_path)

    test_path = paths.raw.test_dataset()
    test_data = pd.read_csv(test_path)

    clean_data(train_data)
    compute_corr(train_data)
    # train_data.to_csv(paths.interim.train_dataset())

    # clean_data(test_data)
    # test_data.to_csv(paths.interim.test_dataset())


if __name__ == "__main__":
    apply()
