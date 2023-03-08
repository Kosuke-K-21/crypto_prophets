
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin

import datetime


def get_data():
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    yesterday_5yearsback = yesterday - datetime.timedelta(days=365*5)

    yfin.pdr_override()

    tickers = ['BTC', 'ETH', 'BNB', 'XRP', 'HEX', 'ADA', 'DOGE', 'SOL']

    df_ind = pd.date_range(start=yesterday_5yearsback, periods=365*5, freq='D')
    df = pd.DataFrame(columns=tickers, index=df_ind)

    for ticker in tickers:
        data = pdr.get_data_yahoo(
            ticker+'-USD', start=yesterday_5yearsback, end=yesterday)
        df[ticker] = data['Adj Close']

    df = df.dropna()
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['index'])
    df.drop('index', axis=1, inplace=True)

    df.to_csv('../frontend/data/original.csv')

    return df
