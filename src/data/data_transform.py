import re

import numpy as np
import pandas as pd

from src.helpers import paths

pd.options.mode.chained_assignment = None  # default='warn'


def one_hot_encode_clm(df, clm):
    # Get one hot encoding
    one_hot = pd.get_dummies(df[clm])
    one_hot.rename(columns=lambda x: "{}_{}".format(clm, x), inplace=True)

    # Drop column as it is now encoded
    df.drop(columns=[clm], inplace=True)

    # Join the encoded df
    return df.join(one_hot)


def one_hot_encode_test(df, catgories, clm):
    # # Get one hot encoding
    # one_hot = pd.get_dummies(df[clm])
    # print(one_hot.head())
    # one_hot['NDF'] = 1
    # one_hot.rename(columns={'NDF': 'country_destination_NDF'})

    # # rest_clms = set(catgories) - set(['country_destination_NDF'])
    # # for x in list(rest_clms):
    # #     one_hot[x] = 0

    # # Drop column as it is now encoded
    # df.drop(columns=[clm], inplace=True)

    # # Join the encoded df
    # # return df.merge(one_hot, left_on='id', right_on='user_id')
    # return pd.concat([df, one_hot], sort=False)
    # # return df.join(one_hot)

    df.drop(columns=[clm], inplace=True)
    df['country_destination_NDF'] = 1
    return df


def one_hot_encode(df):
    clms = [
        'gender',
        'signup_method',
        'signup_flow',
        'language',
        'affiliate_channel',
        'affiliate_provider',
        'first_affiliate_tracked',
        'signup_app',
        'first_device_type',
        'first_browser'
    ]

    for clm in clms:
        df = one_hot_encode_clm(df, clm)
    return df


def transform_data(df):
    df = one_hot_encode(df)
    df = age_categorization(df)
    df = transform_date(df)
    return df


def transform_date(df):
    df['date_account_created'] = pd.to_datetime(df['date_account_created'])
    df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'])

    df = split_date(df, 'date_account_created')
    df = split_date(df, 'timestamp_first_active')
    df['created_less_active'] = (df['date_account_created'] - df['timestamp_first_active']).dt.days
    df.drop(columns=['date_account_created', 'timestamp_first_active'], inplace=True)
    return df


def split_date(df, clm):
    df['day_'+clm] = df['date_account_created'].dt.weekday
    df['month_'+clm] = df['date_account_created'].dt.month
    df['year_'+clm] = df['date_account_created'].dt.year
    return df


def split_label(df):
    labels = df[['id', 'country_destination']].copy()
    df.drop('country_destination', axis=1, inplace=True)
    return df, labels


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
    df = one_hot_encode_clm(df, 'ageGroup')
    return df


def session_aggregate(df):
    # ## Primary Device ##
    # sum secs elapsed
    summed_df = df[['user_id', 'device_type', 'secs_elapsed']]
    summed_df = summed_df.groupby(['user_id', 'device_type'], as_index=False, sort=False)
    summed_df = summed_df.aggregate('sum')

    # get max sum secs elapsed
    max_df = summed_df[['user_id', 'secs_elapsed']]
    max_df = max_df.groupby(['user_id'], as_index=False, sort=False)
    max_df = max_df.aggregate('max')

    # get devices with max secs
    devices_df = summed_df.merge(max_df, on=['user_id', 'secs_elapsed'], how='inner')
    devices_df = devices_df[['user_id', 'device_type']]

    # ## Action Count ##
    # aggregate and combine actions taken columns
    session_actions = df[['user_id', 'action', 'action_type', 'action_detail', 'secs_elapsed']]
    session_actions.secs_elapsed.fillna(0.0, inplace=True)
    session_actions.fillna('', inplace=True)

    # concat Columns into single column
    session_actions['action_conc'] = session_actions['action'] + '_' + \
        session_actions['action_type'] + '_' + session_actions['action_detail']

    # session_actions['action_conc'] = session_actions['action']
    # session_actions['action_conc'] = np.where(session_actions['action_type'],
    #                                           session_actions['action_conc'] +
    #                                           '_' + session_actions['action_type'],
    #                                           session_actions['action_conc'])
    # session_actions['action_conc'] = np.where(session_actions['action_detail'],
    #                                           session_actions['action_conc'] +
    #                                           '_' + session_actions['action_detail'],
    #                                           session_actions['action_conc'])
    # session_actions.drop(columns=['action', 'action_type', 'action_detail'], inplace=True)

    # sum secs elapsed
    summed_df = session_actions[['user_id', 'action_conc', 'secs_elapsed']]
    summed_df = summed_df.groupby(['user_id', 'action_conc'], as_index=False, sort=False)
    summed_df = summed_df.agg({'secs_elapsed': 'sum'})
    actions_data = summed_df.pivot(index='user_id', columns='action_conc', values='secs_elapsed')
    actions_data.fillna(0, inplace=True)

    # rename Columns
    regex = re.compile('[^a-zA-Z_]')
    categories = list(summed_df['action_conc'].drop_duplicates())
    for category in categories:
        new_category = str(category).replace(" ", "_")
        new_category = regex.sub('', new_category)
        new_category = 'action_conc' + '_' + new_category
        actions_data.rename(columns={category: new_category}, inplace=True)

    # concat data into single column
    devices_df = one_hot_encode_clm(devices_df, 'device_type')
    return actions_data.merge(devices_df, on='user_id', how='outer')


def sync_clms(df, df_other):
    for clm, datatype in df_other.items():
        if clm not in df.columns:
            df[clm] = 0

    df[list(df_other.keys())]
    return df


def sync_row(df_res, df_ref):
    df_res = df_ref.merge(df_res, on='id', how='inner')
    df_res = df_res[['id',
                     'country_destination_AU',
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
                     'country_destination_other'
                     ]]
    return df_res


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
    df_train_session = session_aggregate(df_train_session)

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
    df_train_labels = one_hot_encode_clm(df_train_labels, 'country_destination')

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
    df_test_labels = one_hot_encode_test(df_test_labels, train_y_info.keys(), 'country')

    # prepare main CSV features
    print("# # Performing Main Featrue Transformations")
    df_test_main = transform_data(df_test_main)

    # prepare session CSV features
    print("# # Performing Session Featrue Transformations")
    df_test_session = session_aggregate(df_test_session)
    df_test_main = df_test_main.merge(df_test_session, left_on=[
        'id'], right_on=['user_id'], how='inner')

    # make sure test and train features are in sync
    print("# # Performing Sync between Test and Train rows")
    # sync clms
    df_test_main = sync_clms(df_test_main, train_x_info)
    df_test_labels = sync_clms(df_test_labels, train_y_info)
    # sync rows
    df_test_labels = sync_row(df_res=df_test_labels, df_ref=df_test_main)

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
    df_train_main = sync_clms(df_train_main, df_test_x_info)
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
