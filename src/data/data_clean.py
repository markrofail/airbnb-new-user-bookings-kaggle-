import numpy as np
import pandas as pd

from src.helpers import paths


def fix_date(df):
    # Correct Formating
    df['date_account_created'] = pd.to_datetime(df['date_account_created'])
    df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'])

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


def split_seesion(session_df, test_df):
    # get test user ids
    test_user_ids = list(set(test_df['id']))

    # split session according to ids
    session_test_df = session_df[session_df['user_id'].isin(test_user_ids)]
    session_train_df = session_df[~session_df['user_id'].isin(test_user_ids)]

    return session_test_df, session_train_df


def clean_data(df):
    fix_date(df)
    fix_age(df)
    fix_affliate(df)


def apply():
    # ## Train Dataset ##
    train_df = pd.read_csv(paths.raw.train_dataset())
    clean_data(train_df)
    train_df.to_csv(paths.interim.train_dataset(), index=False)

    # ## Tests Dataset ##
    test_df = pd.read_csv(paths.raw.test_dataset())
    clean_data(test_df)
    test_df.to_csv(paths.interim.test_dataset(), index=False)

    # ## Split Session ##
    session_df = pd.read_csv(paths.raw.session_csv())
    session_test_df, session_train_df = split_seesion(session_df, test_df)
    session_train_df.to_csv(paths.interim.session_train(), index=False)
    session_test_df.to_csv(paths.interim.session_test(), index=False)


if __name__ == "__main__":
    print("# Performing Data Clean...")
    apply()
