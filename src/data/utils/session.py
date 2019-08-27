import re

from src.data import utils


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

    # sum secs elapsed
    session_actions['count'] = 1
    summed_df = session_actions[['user_id', 'action_conc', 'count']]
    summed_df = summed_df.groupby(['user_id', 'action_conc'], as_index=False, sort=False)
    summed_df = summed_df.agg({'count': 'sum'})
    actions_data = summed_df.pivot(index='user_id', columns='action_conc', values='count')
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
    devices_df = utils.onehot.one_hot_encode_clm(devices_df, 'device_type')
    return actions_data.merge(devices_df, on='user_id', how='outer')


def split_seesion(session_df, test_df):
    # get test user ids
    test_user_ids = list(set(test_df['id']))

    # split session according to ids
    session_test_df = session_df[session_df['user_id'].isin(test_user_ids)]
    session_train_df = session_df[~session_df['user_id'].isin(test_user_ids)]

    return session_test_df, session_train_df
