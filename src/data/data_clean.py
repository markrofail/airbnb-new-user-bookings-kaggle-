import pandas as pd

from src.helpers import paths
from src.data import utils


def fix_affliate(df):
    # Fill Missing Data
    df['first_affiliate_tracked'].fillna("UKN", inplace=True)


def clean_data(df):
    utils.date.fix_date(df)
    utils.age.fix_age(df)
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
    session_test_df, session_train_df = utils.session.split_seesion(session_df, test_df)
    session_train_df.to_csv(paths.interim.session_train(), index=False)
    session_test_df.to_csv(paths.interim.session_test(), index=False)


if __name__ == "__main__":
    print("# Performing Data Clean...")
    apply()
