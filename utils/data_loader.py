#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import h5py
import logging
from typing import List, Set, Dict, Tuple, Optional, Union
import global_settings

def load_stocks_hist_prices():
    """
    read stocks historical prices from h5
    :return:
    """
    stocks_hist_prices_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                stocks_hist_prices_dict[k] = None
    for k in stocks_hist_prices_dict.keys():
        stocks_hist_prices_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5'), key=k)
    stocks_asofdate = stocks_hist_prices_dict['SPX'].index[-1]
    return stocks_hist_prices_dict, stocks_asofdate


def load_futures_meta_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    read futures meta data from csv
    :return:
    """
    # read_file = os.path.join(global_settings.root_path, 'data/futures_meta.xlsx')
    # futures_meta_data_df = pd.read_excel(read_file, keep_default_na=False, sheet_name='Contracts')
    futures_meta_df = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_meta.csv'), index_col=0)
    futures_meta_df = futures_meta_df[~np.isnan(futures_meta_df['QuandlMultiplier'])]

    futures_contracts_meta_df = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_contract_meta.csv'), index_col=0, keep_default_na=False)
    futures_contracts_meta_df['Last_Trade_Date'] = pd.to_datetime(futures_contracts_meta_df['Last_Trade_Date'])

    inter_comdty_spread_meta_df = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/inter_comdty_spread_meta.csv'), keep_default_na=False)

    inter_comdty_spread_contracts_meta_df = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/inter_comdty_spread_contract_meta.csv'), index_col=0, keep_default_na=False)
    inter_comdty_spread_contracts_meta_df['Last_Trade_Date'] = pd.to_datetime(inter_comdty_spread_contracts_meta_df['Last_Trade_Date'])
    return futures_meta_df, futures_contracts_meta_df, inter_comdty_spread_meta_df, inter_comdty_spread_contracts_meta_df


def load_futures_meta(root: str = None) -> pd.DataFrame:
    """
    get futures contract meta
    :param root: ES, CL, etc
    :return: dataframe of contract meta data
    """
    df_futures_contracts_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_contract_meta.csv'), index_col=0, keep_default_na=False)
    df_futures_contracts_meta['Last_Trade_Date'] = pd.to_datetime(df_futures_contracts_meta['Last_Trade_Date'])
    
    if not root is None:
        try:
            df_futures_contracts_meta = df_futures_contracts_meta[df_futures_contracts_meta.Root==root]
        except:
            pass
    return df_futures_contracts_meta


def load_futures_hist_prices(root: str = None) -> Union[Dict, pd.DataFrame]:
    """
    get futures daily historical price
    :param root: ES, CL, etc
    :return: dataframe or dict of dataframes
    """
    # cache_dir = os.path.dirname(os.path.realpath(__file__))
    if root is None:
        futures_hist_prices_dict = dict()
        if os.path.isfile(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5')):
            with h5py.File(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), 'r') as f:
                for k in f.keys():
                    futures_hist_prices_dict[k] = None
        for k in futures_hist_prices_dict.keys():
            futures_hist_prices_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key=k)
        #futures_asofdate = futures_hist_prices_dict['CL'].index[-1]
        return futures_hist_prices_dict
    else:
        try:
            df = pd.read_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key=root)
        except:
            df = pd.DataFrame()
        return df


def load_inter_comdty_spread_hist_prices() -> Dict:
    inter_comdty_spread_hist_data_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                inter_comdty_spread_hist_data_dict[k] = None
    for k in inter_comdty_spread_hist_data_dict.keys():
        inter_comdty_spread_hist_data_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_historical_prices.h5'), key=k)
    return inter_comdty_spread_hist_data_dict


def load_comdty_generic_hist_prices() -> Dict:
    """
    construct generic prices series on the fly
    :return:
    """
    generic_futures_hist_prices_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/futures_generic_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/futures_generic_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                generic_futures_hist_prices_dict[k] = None
    for k in generic_futures_hist_prices_dict.keys():
        generic_futures_hist_prices_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/futures_generic_historical_prices.h5'), key=k)
    return generic_futures_hist_prices_dict


def load_inter_comdty_generic_hist_prices() -> Dict:
    """
    construct generic prices series on the fly
    :return:
    """
    generic_inter_comdty_hist_prices_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_generic_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_generic_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                generic_inter_comdty_hist_prices_dict[k] = None
    for k in generic_inter_comdty_hist_prices_dict.keys():
        generic_inter_comdty_hist_prices_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_generic_historical_prices.h5'), key=k)
    return generic_inter_comdty_hist_prices_dict


def load_spread_score() -> Dict:
    spread_score_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/spread_scores.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/spread_scores.h5'), 'r') as f:
            for k in f.keys():
                spread_score_dict[k] = None
    for k in spread_score_dict.keys():
        spread_score_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/spread_scores.h5'), key=k)
    return spread_score_dict


def load_fly_score() -> Dict:
    fly_score_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/fly_scores.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/fly_scores.h5'), 'r') as f:
            for k in f.keys():
                fly_score_dict[k] = None
    for k in fly_score_dict.keys():
        fly_score_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/fly_scores.h5'), key=k)
    return fly_score_dict


def load_misc() -> Dict:
    misc_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/misc.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/misc.h5'), 'r') as f:
            for k in f.keys():
                misc_dict[k] = None
    for k in misc_dict.keys():
        try:
            misc_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/misc.h5'), key=k)
        except:
            pass
    return misc_dict
