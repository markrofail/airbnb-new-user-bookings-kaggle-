import pandas as pd

from src.data import utils
from src.helpers import paths

pd.options.mode.chained_assignment = None  # default='warn'


def transform_data(df):
    df = utils.onehot.one_hot_encode(df)
    df = utils.age.age_categorization(df)
    df = utils.date.transform_date(df)
    return df


def split_label(df):
    labels = df[['id', 'country_destination']].copy()
    df.drop('country_destination', axis=1, inplace=True)
    return df, labels


def make_train_data():
    # ## Train Dataset ##
    # load csv into memory
    print("# # Loading CSVs into Memory")
    df_train_main = pd.read_csv(paths.interim.train_dataset())
    df_train_session = pd.read_csv(paths.interim.session_train())

    # prepare main CSV features
    print("# # Performing Main Featrue Transformations")
    df_train_main = transform_data(df_train_main)

    # prepare session CSV features
    print("# # Performing Session Featrue Transformations")
    df_train_session = utils.session.session_aggregate(df_train_session)

    # # sync rows
    # print("# # Performing Sync between Main and Session rows")
    # df_train_labels = sync_row(df_res=df_train_labels, df_ref=df_train_main)

    # merge CSVs and save
    print("# # Merge Session and Main CSVs and save")
    df_train_main = df_train_main.merge(df_train_session, left_on=[
                                        'id'], right_on=['user_id'], how='inner')

    # split and prepare labels
    print("# # Spliting Labels")
    df_train_main, df_train_labels = split_label(df_train_main)
    # df_train_labels = one_hot_encode_clm(df_train_labels, 'country_destination')

    df_train_main.to_csv(paths.processed.train_dataset(), index=False)
    df_train_labels.to_csv(paths.processed.train_labels(), index=False)

    return (dict(zip(df_train_main.columns, df_train_main.dtypes)),
            dict(zip(df_train_labels.columns, df_train_labels.dtypes)))


def make_test_data(train_x_info, train_y_info):
    # ## Test Dataset ##
    # load csv into memory
    print("# # Loading CSVs into Memory")
    df_test_main = pd.read_csv(paths.interim.test_dataset())
    df_test_session = pd.read_csv(paths.interim.session_test())

    # split and prepare labels
    print("# # Spliting Labels")
    df_test_labels = pd.read_csv(paths.raw.test_labels())
    df_test_labels = utils.onehot.one_hot_encode_test(
        df_test_labels, train_y_info.keys(), 'country')

    # prepare main CSV features
    print("# # Performing Main Featrue Transformations")
    df_test_main = transform_data(df_test_main)

    # prepare session CSV features
    print("# # Performing Session Featrue Transformations")
    df_test_session = utils.session.session_aggregate(df_test_session)
    df_test_main = df_test_main.merge(df_test_session, left_on=[
        'id'], right_on=['user_id'], how='inner')

    # make sure test and train features are in sync
    print("# # Performing Sync between Test and Train rows")
    # sync clms
    df_test_main = utils.sync.sync_clms(df_test_main, train_x_info)
    df_test_labels = utils.sync.sync_clms(df_test_labels, train_y_info)
    # sync rows
    df_test_labels = utils.sync.sync_row(df_res=df_test_labels, df_ref=df_test_main)

    # merge CSVs and save
    print("# # Merge Session and Main CSVs and save")
    df_test_main.to_csv(paths.processed.test_dataset(), index=False)
    df_test_labels.to_csv(paths.processed.test_labels(), index=False)

    return dict(zip(df_test_main.columns, df_test_main.dtypes))


if __name__ == "__main__":
    print("# Performing TRAIN Data Transformation...")
    df_train_x_info, df_train_y_info = make_train_data()

    print("\n# Performing TEST Data Transformation...")
    df_test_x_info = make_test_data(df_train_x_info, df_train_y_info)

    print("\n# Performing Sync between Test and Train features...")
    df_train_main = pd.read_csv(paths.processed.train_dataset())
    df_train_main = utils.sync.sync_clms(df_train_main, df_test_x_info)
    paths.processed.train_dataset().unlink()
    df_train_main.to_csv(paths.processed.train_dataset(), index=False)

    print("# ========================")
    X_train = pd.read_csv(paths.processed.train_dataset())
    Y_train = pd.read_csv(paths.processed.train_labels())
    X_test = pd.read_csv(paths.processed.test_dataset())
    Y_test = pd.read_csv(paths.processed.test_labels())
    print(X_train.shape, X_test.shape)
    print(Y_train.shape, Y_test.shape)
    print("# ========================")
