import pandas as pd


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


def fix_date(df):
    # Correct Formating
    df['date_account_created'] = pd.to_datetime(df['date_account_created'])
    df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'])

    # Remove Non essential data
    df.drop('date_first_booking', axis=1, inplace=True)
