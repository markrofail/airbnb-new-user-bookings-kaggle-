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
