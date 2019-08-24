import numpy as np
import pandas as pd

from src.helpers import paths


def find_nulls(df):
    print("# Columns with Null Records:")
    for clm in df.columns:
        null_count = df[clm].isnull().sum()
        if null_count > 0:
            print("\t\"{}\": {} records".format(clm, null_count))


if __name__ == "__main__":
    train_path = paths.raw.train_dataset()
    train_data = pd.read_csv(train_path)
    find_nulls(train_data)

    train_path = paths.interim.train_dataset()
    train_data = pd.read_csv(train_path)
    find_nulls(train_data)

    # test_path = paths.raw.test_dataset()
    # test_data = pd.read_csv(test_path)
    # find_nulls(test_data)

    # test_path = paths.interim.test_dataset()
    # test_data = pd.read_csv(test_path)
    # find_nulls(test_data)
