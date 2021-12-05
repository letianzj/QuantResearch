#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Set, Dict, Tuple, Optional
from scipy import stats
import requests
import quandl
import h5py
import logging
from barchart_ondemand import OnDemandClient
from futures_tools import get_futures_chain, get_generic_futures_hist_data, get_futures_generic_ticker
import global_settings

def download_generic_futures_hist_prices_from_quandl() -> None:
    pass


def download_futures_hist_prices_from_quandl() -> None:
    """
    :return:
    """
    # start_date = datetime(2017, 1, 1)
    end_date = datetime.today()
    start_date = end_date + timedelta(days=-75)

    df_futures_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_meta.csv'), index_col=0)
    df_futures_meta = df_futures_meta[~np.isnan(df_futures_meta['QuandlMultiplier'])]
    df_futures_contracts_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_contract_meta.csv'), index_col=0, keep_default_na=False)
    df_futures_contracts_meta = df_futures_contracts_meta[~df_futures_contracts_meta['Last_Trade_Date'].isnull()]       # remove empty last_trade_date; same as keep_default_na=False
    df_futures_contracts_meta['Last_Trade_Date'] = pd.to_datetime(df_futures_contracts_meta['Last_Trade_Date'])

    futures_hist_prices_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                futures_hist_prices_dict[k] = None

    for row_idx, row in df_futures_meta.iterrows():
        quandl_ticker = row['Quandl']
        quandl_multiplier = row['QuandlMultiplier']
        if not isinstance(quandl_ticker, str):             # empty is type(np.nan) == float
            continue
        if quandl_multiplier == 0:
            continue

        # download new dataset, combine with old dataset
        df_hist_prices = pd.DataFrame()
        try:
            # find all eligible contracts
            df_futures_contract_meta = df_futures_contracts_meta[df_futures_contracts_meta['Root'] == row_idx].copy()
            df_futures_contract_meta.sort_values('Last_Trade_Date', inplace=True)
            df_futures_contract_meta = get_futures_chain(df_futures_contract_meta,  start_date)

            for row_idx2, row2 in df_futures_contract_meta.iterrows():
                if row_idx == 'UX':      # directly from CBOE
                    try:
                        # https://markets.cboe.com/us/futures/market_statistics/historical_data/
                        url = fr'https://markets.cboe.com/us/futures/market_statistics/historical_data/products/csv/VX/{row2["Last_Trade_Date"].strftime("%Y-%m-%d")}/'
                        r = requests.get(url, stream=True)
                        data = r.content.decode('utf8')
                        df = pd.read_csv(io.StringIO(data))
                        df.set_index('Trade Date', inplace=True)
                        df = df['Settle']
                        df.name = row_idx2
                        df.index = pd.to_datetime(df.index)
                        df.sort_index(ascending=True, inplace=True)
                        df_hist_prices = pd.concat([df_hist_prices, df], axis=1, join='outer', sort=True)

                        logging.debug('Contract {} is downloaded'.format(row_idx2))
                    except:
                        logging.error('Contract {} is missing'.format(row_idx2))
                else:
                    try:
                        quandl_contract = quandl_ticker[:-5] + row_idx2[-5:]
                        df = quandl.get(quandl_contract, start_date=start_date, end_date=end_date,
                                        qopts={'columns': ['Settle']}, authtoken=global_settings.quandl_auth)
                        try:
                            df = df['Settle']
                        except:
                            df = df['Last']
                            df.name = 'Settle'
                        df.name = row_idx2
                        if not np.isnan(quandl_multiplier):         # consistent with Bloomberg
                            df = df * quandl_multiplier
                        df_hist_prices = pd.concat([df_hist_prices, df], axis=1, join='outer', sort=True)

                        logging.debug('Contract {} is downloaded'.format(row_idx2))
                    except:
                        logging.error('Contract {} is missing'.format(row_idx2))

                time.sleep(3)

            # update existing dataset
            if row_idx in futures_hist_prices_dict.keys():
                df_old = pd.read_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key=row_idx)
                df_hist_prices = df_hist_prices.combine_first(df_old)

            df_hist_prices.sort_index(inplace=True)
            df_hist_prices.to_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key=row_idx)
            logging.debug('{} is download'.format(row_idx))
        except:
            logging.error('{} failed to download'.format(row_idx))


def download_futures_hist_prices_from_barchart(grps) -> None:
    """
    Up to 6 months of daily history
    Up to 150 queries per day
    :return:
    """
    # start_date = datetime(2000, 1, 1)
    end_date = datetime.today()
    start_date = end_date + timedelta(days=-180)

    od = OnDemandClient(api_key=global_settings.barchart_auth, end_point='https://marketdata.websol.barchart.com/')

    df_futures_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_meta.csv'), index_col=0, keep_default_na=False)
    df_futures_meta = df_futures_meta[df_futures_meta['Barchart'] != '']
    df_futures_contracts_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_contract_meta.csv'), keep_default_na=False)
    df_futures_contracts_meta['Last_Trade_Date'] = pd.to_datetime(df_futures_contracts_meta['Last_Trade_Date'])

    futures_hist_prices_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                futures_hist_prices_dict[k] = None
    for k in futures_hist_prices_dict.keys():
        futures_hist_prices_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key=k)

    for row_idx, row in df_futures_meta.iterrows():
        if int(row['BarchartGroup']) not in grps:
            continue
        logging.info('downloading ' + row_idx)
        sym = row_idx
        bar_sym = row['Barchart']
        # get all non-expired contracts
        valid_contracts = df_futures_contracts_meta.loc[df_futures_contracts_meta['Root'] == sym].copy()
        valid_contracts.sort_values(by=['Last_Trade_Date'], inplace=True)
        valid_contracts = valid_contracts[valid_contracts['Last_Trade_Date'] > start_date]
        df_sym = pd.DataFrame()
        for _, row_c in valid_contracts.iterrows():
            c = row_c['Contract']  # BBG symbol
            try:
                cb = bar_sym + c[2] + c[-2:]               # barchart symbol
                resp = od.history(cb, 'daily', startDate = start_date.strftime('%Y%m%d'), maxRecords = 500)
                df = pd.DataFrame(resp['results'])
                df = df.set_index('tradingDay')
                df.index = pd.to_datetime(df.index)
                df.index = df.index.date
                df = df['close']
                df.name = c
                df_sym = pd.concat([df_sym, df], join='outer', axis=1, sort=True)
                logging.info(c+ ' downloaded')
            except:
                logging.info(c + ' skipped')

        # update existing dataset
        if row_idx in futures_hist_prices_dict.keys():
            df_old = pd.read_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key=k)
            df_sym = df_sym.combine_first(df_old)

        df_sym.sort_index(inplace=True)
        df_sym.to_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key=row_idx)
        logging.debug('{} is download'.format(row_idx))

def download_vix_futures_from_cboe():
    df_old = None
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), 'r') as f:
            if 'UX' in f.keys():
                df_old = pd.read_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key='UX')

    end_date = datetime.today()
    date_list = [end_date - timedelta(days=x) for x in range(global_settings.lookback_days)]
    df_vix = pd.DataFrame()
    for asofdate in date_list:
        datestr = asofdate.strftime('%Y-%m-%d')
        url = fr'https://markets.cboe.com/us/futures/market_statistics/settlement/csv?dt={datestr}'
        # params = {'dt': datestr}
        # r = requests.post(url, data=params)
        r = requests.get(url, stream=True)
        if r.ok:
            data = r.content.decode('utf8')
            df = pd.read_csv(io.StringIO(data))
            df.set_index('Product', inplace=True)
            df = df[df.index == 'VX']
            df = df[df['Symbol'].str.contains('VX/')]
            df_row = df[['Price']]    # dataframe
            df_row.index = [row['Symbol'].replace('VX/', 'UX')[:-1]+row['Expiration Date'][:4] for idx, row in df.iterrows()]
            df_row = df_row.transpose()
            df_row.index =[asofdate.strftime('%Y-%m-%d')]
            df_vix = df_row.combine_first(df_vix)

        time.sleep(1)
        print('VIX ' + asofdate.strftime("%Y-%m-%d") + ' is done')

    df_vix.index = pd.to_datetime(df_vix.index)
    df_vix.sort_index(inplace=True)
    df_vix.dropna(axis=0, how='all', inplace=True)
    # update existing dataset
    if df_old:
        df_vix = df_vix.combine_first(df_old)

    df_vix.to_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key='UX')
