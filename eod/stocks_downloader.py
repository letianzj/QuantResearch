#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Set, Dict, Tuple, Optional
import yfinance as yf
import h5py
import csv
import requests
import logging
import global_settings


def download_stocks_hist_prices() -> None:
    """
        download stocks historical prices from IEX
        and save to h5
        :return:
    """
    # cache_dir = os.path.dirname(os.path.realpath(__file__))
    # start_date = datetime(2000, 1, 1)
    end_date = datetime.today()
    # start_date = end_date.replace(year=end_date.year - 5)  # restriction from IEX
    start_date = end_date + timedelta(days=-global_settings.lookback_days)

    # df_stocks_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/stocks_meta.csv'), index_col=0, keep_default_na=False)
    # df_stocks_meta = df_stocks_meta[df_stocks_meta['YAHOO'] != '']
    df_stocks_meta = pd.DataFrame.from_dict({
        'DXY': 'DX-Y.NYB',         # futures on NYBOT
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'USDCAD': 'USDCAD=X',
        'USDCNY': 'USDCNY=X',
        'BTC': 'BTC-USD',
        'SPX': '^GSPC',
        'NDX': '^IXIC',
        'RUT': '^RUT',
        'VIX': '^VIX'
    }, orient='index', columns=['YAHOO'])

    stocks_hist_prices_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path,'data/stocks_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                stocks_hist_prices_dict[k] = None

    logging.info('Start downloading stock data')
    for row_idx, row in df_stocks_meta.iterrows():
        try:
            #df = pdr.DataReader(name=row['YAHOO'], data_source='yahoo', start=start_date, end=end_date)
            df = yf.download(row['YAHOO'], start=start_date, end=end_date)
            if row_idx in stocks_hist_prices_dict.keys():
                df_old = pd.read_hdf(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5'), key=row_idx)
                df = df.combine_first(df_old)

            df.sort_index(inplace=True)
            df.to_hdf(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5'), key=row_idx)

            logging.info('{} is downloaded'.format(row_idx))
            time.sleep(1)
        except:
            logging.error('{} failed to download'.format(row_idx))


def download_stocks_hist_1m_data() -> None:
    # end_date = datetime(2014, 11, 1)
    end_date = datetime.today()
    nlookback = -5         # yahoo limit to 30d

    df_stocks_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/intraday_stocks.csv'), header=None)

    stocks_hist_intraday_data_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/stocks_historical_intraday_data.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/stocks_historical_intraday_data.h5'), 'r') as f:
            for k1 in f.keys():
                stocks_hist_intraday_data_dict[k1] = None
                for k2 in f[k1].keys():
                    stocks_hist_intraday_data_dict[k1][k2] = None
    # for k1 in stocks_hist_intraday_data_dict.keys():
    #     for k2 in stocks_hist_intraday_data_dict[k1].keys():
    #         stocks_hist_intraday_data_dict[k1][k2] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/stocks_historical_intraday_data.h5'), key=f'{k1}/{k2}')

    logging.info('Start downloading stock stock intraday data')
    for i in range(nlookback, 0, 1):
        sd = end_date + timedelta(days=i)
        # sd = datetime(2020, 7, 6)
        ed = sd + timedelta(days=1)
        print(sd, ed)

        for _, row in df_stocks_meta.iterrows():
            sym = row.iloc[0]
            if sym == 'SPY':
                print('downloading SPY...')
            try:
                if sym not in stocks_hist_intraday_data_dict.keys():
                    stocks_hist_intraday_data_dict[sym] = dict()
                if sd.date() in stocks_hist_intraday_data_dict[sym].keys():       # already saved
                    if sym == 'SPY':
                        print('SPY exists', sd)
                    continue
                df = yf.download(tickers=sym, start=sd, end=ed, interval="1m")
                if (df.shape[0] == 0) and (sym == 'SPY'):      # not a valid date; according to SPY
                    print('SPY empty')
                    break
                if df.shape[0] == 0:           # nothing to record
                    continue
                if (df.index[0].hour != 9) and (df.index[0].minute != 30):      # corrupted
                    if sym == 'SPY':
                        print('SPY start time failed', df.index[0])
                    continue
                if ((df.index[-1].hour != 15) and (df.index[-1].minute != 59)) and ((df.index[-1].hour != 12) and (df.index[-1].minute != 59)):      # corrupted
                    df = df.iloc[:-1, :]      # might have next 9:30
                    if ((df.index[-1].hour != 15) and (df.index[-1].minute != 59)) and ((df.index[-1].hour != 12) and (df.index[-1].minute != 59)):    
                        if sym == 'SPY':
                           print('SPY end time failed', df.index[-1])
                        continue
                dt = df.index[0].to_pydatetime().date()
                df.to_hdf(os.path.join(global_settings.root_path, 'data/stocks_historical_intraday_data.h5'), key=f'{sym}/{dt}')

                logging.info('{} intraday is downloaded'.format(sym))
                time.sleep(5)
            except:
                logging.error('{} intraday failed to download'.format(sym))


def download_fx_rates_from_ecb() -> None:
    ecb_dict = {'FX:EURUSD': 'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/usd.xml',
                'FX:GBPUSD': 'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/gbp.xml',
                'FX:USDJPY': 'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/jpy.xml',
                'FX:USDCAD': 'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/cad.xml',
                'FX:USDCNY': 'https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/cny.xml'}

    stocks_hist_prices_dict = dict()
    for k in ecb_dict.keys():
        try:
            stocks_hist_prices_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5'), key=k)
        except:
            stocks_hist_prices_dict[k] = pd.DataFrame()

    for k, v in ecb_dict.items():
        try:
            df = pd.DataFrame(columns=['Close'])
            url = v
            response = requests.get(url)
            data = response.content.decode('utf8')
            for row in csv.reader(data.split('\n'), delimiter=',', quotechar='"'):
                if 'TIME_PERIOD' not in row[0]:
                    continue
                dt = row[0].split(' ')[1]
                idx1 = dt.find('"')
                idx2 = dt.rfind('"')
                dt = datetime.strptime(dt[idx1+1:idx2], '%Y-%m-%d')
                dv = row[0].split(' ')[2]
                idx1 = dv.find('"')
                idx2 = dv.rfind('"')
                dv = float(dv[idx1+1:idx2])
                df.loc[dt]= dv

            df.sort_index(inplace=True)
            if k == 'FX:EURUSD':
                stocks_hist_prices_dict[k] = df.combine_first(stocks_hist_prices_dict[k])
            elif k in ['FX:GBPUSD']:  # EURUSD / EURGBP; at this time EURUSD is known
                df2 = pd.merge(df, stocks_hist_prices_dict['FX:EURUSD'], how='inner', left_index=True, right_index=True)
                df3 = df2.iloc[:, [1]] / df2.iloc[:, [0]]
                stocks_hist_prices_dict[k] = df3.combine_first(stocks_hist_prices_dict[k])
            else:  # USDJPY = EURJPY / EURUSD
                df2 = pd.merge(df, stocks_hist_prices_dict['FX:EURUSD'], how='inner', left_index=True, right_index=True)
                df3 = df2.iloc[:, [0]] / df2.iloc[:, [1]]
                stocks_hist_prices_dict[k] = df3.combine_first(stocks_hist_prices_dict[k])

            stocks_hist_prices_dict[k].sort_index(inplace=True)
            stocks_hist_prices_dict[k].to_hdf(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5'), key=k)
            logging.info('{} is downloaded'.format(k))
        except:
            logging.info('{} download failed'.format(k))
        time.sleep(1)


def download_vix_index_from_cboe() -> None:
    url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv'
    # url = 'http://www.cboe.com/delayedquote/detailed-quotes?ticker=%5eVIX'
    webpage = requests.get(url)
    data = webpage.content.decode('utf8')
    data = data[data.index('Date,VIX'):]
    df = pd.read_csv(io.StringIO(data))
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.columns = [c.replace('VIX ', '') for c in df.columns]
    df['Adj Close'] = df['Close']
    df['Volume'] = 0.0

    df.to_hdf(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5'), key='VIX')
