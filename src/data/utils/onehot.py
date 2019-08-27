import pandas as pd


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
