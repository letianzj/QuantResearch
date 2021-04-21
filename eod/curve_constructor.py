#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import h5py
import logging
import global_settings
import data_loader
from futures_tools import get_futures_chain, get_futures_actual_ticker, get_generic_futures_hist_data

def construct_inter_commodity_spreads() -> None:
    #read_file = os.path.join(global_settings.root_path, 'data/futures_meta.xlsx')
    #df_config = pd.read_excel(read_file, keep_default_na=False, sheet_name='Spread')
    #df_contracts = pd.read_excel(read_file, keep_default_na=False, sheet_name='Contracts')
    df_config = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/inter_comdty_spread_meta.csv'), keep_default_na=False)
    df_futures_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_meta.csv'), index_col=0, keep_default_na=False)
    df_futures_contracts_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_contract_meta.csv'), index_col=0, keep_default_na=False)
    df_futures_contracts_meta['Last_Trade_Date'] = pd.to_datetime(df_futures_contracts_meta['Last_Trade_Date'])
    df_futures_contracts_meta = df_futures_contracts_meta.groupby('Root')

    start_year = 2000
    end_year = datetime.today().year + 30  # 30 year, to be conservative

    futures_data_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                futures_data_dict[k] = None
    for k in futures_data_dict.keys():
        futures_data_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key=k)

    inter_comdty_spread_hist_data_dict = {}
    meta_data_spread = pd.DataFrame(columns=['First_Trade_Date', 'Last_Trade_Date'])
    for _, row in df_config.iterrows():
        Leg1 = row['Leg1'] + ' ' if len(row['Leg1']) == 1 else row['Leg1']
        Leg2 = row['Leg2'] + ' ' if len(row['Leg2']) == 1 else row['Leg2']
        Leg3 = row['Leg3'] + ' ' if len(row['Leg3']) == 1 else row['Leg3']
        weight1 = float(row['Weight1'])
        weight2 = float(row['Weight2'])
        gen_month1 = df_futures_meta.loc[Leg1, 'FUT_GEN_MONTH']
        gen_month2 = df_futures_meta.loc[Leg2, 'FUT_GEN_MONTH']
        common_months = list(set(gen_month1).intersection(set(gen_month2)))
        sym_root = '{}:{}:{}:{}:'.format(Leg1, Leg2, round(weight1, 4), round(weight2, 4))

        hist_data_spread = pd.DataFrame()
        if Leg3:        #if isinstance(row['Leg3'], str):
            weight3 = float(row['Weight3'])
            sym_root = '{}:{}:{}:{}:{}:{}:'.format(Leg1, Leg2, Leg3, round(weight1, 4), round(weight2, 4), round(weight3, 4))
            gen_month3 = df_futures_meta.loc[Leg3, 'FUT_GEN_MONTH']
            common_months = list(set(common_months).intersection(set(gen_month3)))

        # assemble common month, meta_data is the earliest last_trading_day
        # name is A:B:C:w1:w2:w3:J06
        for yr in range(start_year, end_year + 1):
            for mth in common_months:
                exist = (Leg1 + mth + str(yr) in df_futures_contracts_meta.get_group(Leg1).index) and (
                            Leg2 + mth + str(yr) in df_futures_contracts_meta.get_group(Leg2).index)
                if Leg3:
                    exist = exist and (Leg3 + mth + str(yr) in df_futures_contracts_meta.get_group(Leg3).index)

                if not exist:
                    continue

                try:
                    row_dict = {}
                    s = futures_data_dict[Leg1][Leg1 + mth + str(yr)] * weight1 \
                        + futures_data_dict[Leg2][Leg2 + mth + str(yr)] * weight2
                    if Leg3:
                        s = s + futures_data_dict[Leg3][Leg3 + mth + str(yr)] * weight3
                        # row_dict['First_Trade_Date'] = max(
                        #     df_futures_contracts_meta.get_group(Leg1).loc[Leg1 + mth + str(yr), 'First_Trade_Date'],
                        #     df_futures_contracts_meta.get_group(Leg2).loc[Leg2 + mth + str(yr), 'First_Trade_Date'],
                        #     df_futures_contracts_meta.get_group(Leg3).loc[Leg3 + mth + str(yr), 'First_Trade_Date'])
                        row_dict['Last_Trade_Date'] = min(
                            df_futures_contracts_meta.get_group(Leg1).loc[Leg1 + mth + str(yr), 'Last_Trade_Date'],
                            df_futures_contracts_meta.get_group(Leg2).loc[Leg2 + mth + str(yr), 'Last_Trade_Date'],
                            df_futures_contracts_meta.get_group(Leg3).loc[Leg3 + mth + str(yr), 'Last_Trade_Date'])
                    else:
                        # row_dict['First_Trade_Date'] = max(
                        #     df_futures_contracts_meta.get_group(Leg1).loc[Leg1 + mth + str(yr), 'First_Trade_Date'],
                        #     df_futures_contracts_meta.get_group(Leg2).loc[Leg2 + mth + str(yr), 'First_Trade_Date'])
                        row_dict['Last_Trade_Date'] = min(
                            df_futures_contracts_meta.get_group(Leg1).loc[Leg1 + mth + str(yr), 'Last_Trade_Date'],
                            df_futures_contracts_meta.get_group(Leg2).loc[Leg2 + mth + str(yr), 'Last_Trade_Date'])

                    row_dict['Root'] = sym_root
                    s.name = sym_root + mth + str(yr)
                    hist_data_spread = pd.concat([hist_data_spread, s], axis=1, sort=True)
                    df_2 = pd.DataFrame(row_dict, index=[s.name])
                    meta_data_spread = meta_data_spread.append(df_2)
                    logging.debug('{} {} {} constructed'.format(sym_root, yr, mth))
                except:
                    logging.debug('{} {} {} passed'.format(sym_root, yr, mth))

        inter_comdty_spread_hist_data_dict[sym_root] = hist_data_spread
        inter_comdty_spread_hist_data_dict[sym_root].to_hdf(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_historical_prices.h5'), key=sym_root)
        logging.info('{} is constructed'.format(sym_root))

    meta_data_spread.sort_values(by='Last_Trade_Date', inplace=True,  axis=0, ascending=True)
    meta_data_spread.to_csv(os.path.join(global_settings.root_path, 'data/config/inter_comdty_spread_contract_meta.csv'), index=True)
    logging.info('commodity_inter_spread saved')


def construct_comdty_generic_hist_prices() -> None:
    """
    construct generic prices series on the fly
    :return:
    """
    generic_futures_hist_prices_dict = {}

    df_futures_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_meta.csv'), index_col=0)
    df_futures_meta = df_futures_meta[~np.isnan(df_futures_meta['QuandlMultiplier'])]
    df_futures_contracts_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_contract_meta.csv'), index_col=0, keep_default_na=False)
    df_futures_contracts_meta['Last_Trade_Date'] = pd.to_datetime(df_futures_contracts_meta['Last_Trade_Date'])
    df_futures_contracts_meta = df_futures_contracts_meta.groupby('Root')

    futures_data_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                futures_data_dict[k] = None
    for k in futures_data_dict.keys():
        futures_data_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), key=k)

    for idx, _ in df_futures_meta.iterrows():
        root_sym = idx
        try:
            gen = get_generic_futures_hist_data(futures_data_dict[root_sym], df_futures_contracts_meta.get_group(root_sym))
            generic_futures_hist_prices_dict[root_sym] = gen
            generic_futures_hist_prices_dict[root_sym].to_hdf(os.path.join(global_settings.root_path, 'data/futures_generic_historical_prices.h5'), key=root_sym)
            logging.info('{} generic prices generated'.format(root_sym))
        except:
            logging.error('{} failed to generate generic prices'.format(root_sym))


def construct_inter_comdty_generic_hist_prices() -> None:
    """
    construct generic prices series on the fly
    :return:
    """
    generic_inter_comdty_hist_prices_dict = {}

    df_futures_contracts_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/inter_comdty_spread_contract_meta.csv'), index_col=0, keep_default_na=False)
    df_futures_contracts_meta['Last_Trade_Date'] = pd.to_datetime(df_futures_contracts_meta['Last_Trade_Date'])
    df_futures_contracts_meta = df_futures_contracts_meta.groupby('Root')

    inter_comdty_spread_hist_data_dict = dict()
    if os.path.isfile(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_historical_prices.h5')):
        with h5py.File(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_historical_prices.h5'), 'r') as f:
            for k in f.keys():
                inter_comdty_spread_hist_data_dict[k] = None
    for k in inter_comdty_spread_hist_data_dict.keys():
        inter_comdty_spread_hist_data_dict[k] = pd.read_hdf(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_historical_prices.h5'), key=k)

    for root_sym, group in df_futures_contracts_meta:
        try:
            # gen = get_generic_futures_hist_data(inter_comdty_spread_hist_data_dict[root_sym], df_futures_contracts_meta.get_group(root_sym))
            gen = get_generic_futures_hist_data(inter_comdty_spread_hist_data_dict[root_sym], group)
            generic_inter_comdty_hist_prices_dict[root_sym] = gen
            generic_inter_comdty_hist_prices_dict[root_sym].to_hdf(os.path.join(global_settings.root_path, 'data/inter_comdty_spread_generic_historical_prices.h5'), key=root_sym)
            logging.info('{} generic prices generated'.format(root_sym))
        except:
            logging.error('{} failed to generate generic prices'.format(root_sym))


def construct_curve_spread_fly():
    # cache_dir = os.path.dirname(os.path.realpath(__file__))
    _, futures_contracts_meta_df, _, inter_comdty_spread_contracts_meta_df = data_loader.load_futures_meta_data()
    futures_hist_prices_dict, _ = data_loader.load_futures_hist_prices()
    generic_futures_hist_prices_dict = data_loader.load_comdty_generic_hist_prices()
    inter_comdty_spread_hist_data_dict = data_loader.load_inter_comdty_spread_hist_prices()
    generic_inter_comdty_hist_prices_dict = data_loader.load_inter_comdty_generic_hist_prices()
    
    combined_root_syms = list(generic_futures_hist_prices_dict.keys())
    combined_root_syms.extend(list(generic_inter_comdty_hist_prices_dict.keys()))

    # get spread/fly for outright and inter-comdty-spread
    for sym_root in combined_root_syms:
        if ':' in sym_root:
            hist_data = inter_comdty_spread_hist_data_dict[sym_root]
            meta_data = inter_comdty_spread_contracts_meta_df[inter_comdty_spread_contracts_meta_df['Root'] == sym_root]
            meta_data.sort_values('Last_Trade_Date', inplace=True)
            generic_data = generic_inter_comdty_hist_prices_dict[sym_root]
        else:
            hist_data = futures_hist_prices_dict[sym_root]
            meta_data = futures_contracts_meta_df[futures_contracts_meta_df['Root'] == sym_root]
            meta_data.sort_values('Last_Trade_Date', inplace=True)
            generic_data = generic_futures_hist_prices_dict[sym_root]

        try:
            asofdate = hist_data.index[-1]
        except:     # probably no data
            continue

        meta_data = get_futures_chain(meta_data, asofdate)

        # get spread combos
        spread_combos = []
        tenors = range(1, generic_data.shape[1] + 1)
        for i in tenors:
            for j in tenors:
                spread = j - i
                if i <= 24 and j > i and spread <= 12:
                    spread_combos.append((i, j))

        fly_combos = []
        tenors = range(1, generic_data.shape[1] + 1)
        for i in tenors:
            for j in tenors:
                spread1 = j - i
                for k in tenors:
                    spread2 = k - j
                    if i <= 24 and j > i and k > j and spread1 <= 12 and spread2 <= 12 and spread1 == spread2:
                        fly_combos.append((i, j, k,))

        cols_spread = ['Name', 'Leg1', 'Leg2', 'Leg1 Actual', 'Leg2 Actual', 'Spread', 'Spread Prcnt', 'RD Prcnt', 'Spread Z-Score', 'RD Z-Score']
        df_spread_stats = pd.DataFrame(columns=cols_spread)
        for i in range(len(spread_combos)):
            row_dict = {}
            # extract individual CM time series tickers for combo [i]
            try:
                legA = generic_data[sym_root + str(spread_combos[i][0])]
                legB = generic_data[sym_root + str(spread_combos[i][1])]
            except:
                logging.error('{} {} skipped'.format(sym_root, spread_combos[i]))
                continue

            try:
                legA_RD = generic_data[sym_root + str(spread_combos[i][0] - 1)]
                legB_RD = generic_data[sym_root + str(spread_combos[i][1] - 1)]
                merged = pd.concat([legA, legB, legA_RD, legB_RD], axis=1).dropna(axis=0, how='any')
            except:     # front month has no roll down
                legA_RD = None
                legB_RD = None
                merged = pd.concat([legA, legB], axis=1).dropna(axis=0, how='any')

            try:
                merged['SpreadLevel'] = merged.iloc[:, 0] - merged.iloc[:, 1]
                current_spread_level = merged.iloc[-1]['SpreadLevel']
                percentile_spread = stats.percentileofscore(merged['SpreadLevel'], current_spread_level, kind='mean')
                stdev_pread = np.std(merged['SpreadLevel'])
                mean_spread = np.average(merged['SpreadLevel'])
                z_spread = (current_spread_level - mean_spread) / stdev_pread
                if legA_RD is not None:
                    merged['RolledDownLevel'] = merged.iloc[:, 2] - merged.iloc[:, 3]
                    percentile_RD = stats.percentileofscore(merged['RolledDownLevel'], current_spread_level, kind='mean')
                    stdev_RD = np.std(merged['RolledDownLevel'])
                    mean_RD = np.average(merged['RolledDownLevel'])
                    z_RD = (current_spread_level - mean_RD) / stdev_RD
                else:
                    percentile_RD = np.NaN
                    z_RD = np.NaN

                row_dict['Name'] = sym_root
                row_dict['Leg1'] = spread_combos[i][0]
                row_dict['Leg2'] = spread_combos[i][1]
                row_dict['Leg1 Actual'] = get_futures_actual_ticker(meta_data, legA.name)
                row_dict['Leg2 Actual'] = get_futures_actual_ticker(meta_data, legB.name)
                row_dict['Spread'] = round(current_spread_level, 4)
                row_dict['Spread Prcnt'] = round(percentile_spread, 4)
                row_dict['RD Prcnt'] = round(percentile_RD, 4)
                row_dict['Spread Z-Score'] = round(z_spread, 4)
                row_dict['RD Z-Score'] = round(z_RD, 4)

                df_2 = pd.DataFrame(row_dict, index=[spread_combos[i]])
                df_spread_stats = df_spread_stats.append(df_2)
                logging.info('spread {} {} finished'.format(sym_root, spread_combos[i]))
            except:
                logging.error('spread {} {} failed'.format(sym_root, spread_combos[i]))

        df_spread_stats.to_hdf(os.path.join(global_settings.root_path, 'data/spread_scores.h5'), key=sym_root)

        cols_fly = ['Name', 'Leg1', 'Leg2', 'Leg3', 'Leg1 Actual', 'Leg2 Actual', 'Leg3 Actual', 'Fly', 'Fly Prcnt', 'RD Prcnt',
                       'Fly Z-Score', 'RD Z-Score']
        df_fly_stats = pd.DataFrame(columns=cols_fly)
        for i in range(len(fly_combos)):
            row_dict = {}
            # extract individual CM time series tickers for combo [i]
            try:
                legA = generic_data[sym_root + str(fly_combos[i][0])]
                legB = generic_data[sym_root + str(fly_combos[i][1])]
                legC = generic_data[sym_root + str(fly_combos[i][2])]
            except:
                logging.error('{} {} skipped'.format(sym_root, fly_combos[i]))
                continue

            try:
                legA_RD = generic_data[sym_root + str(fly_combos[i][0] - 1)]
                legB_RD = generic_data[sym_root + str(fly_combos[i][1] - 1)]
                legC_RD = generic_data[sym_root + str(fly_combos[i][2] - 1)]
                merged = pd.concat([legA, legB, legC, legA_RD, legB_RD, legC_RD], axis=1).dropna(axis=0, how='any')
            except:     # front month has no roll down
                legA_RD = None
                legB_RD = None
                legC_RD = None
                merged = pd.concat([legA, legB, legC], axis=1).dropna(axis=0, how='any')

            try:
                merged['FlyLevel'] = merged.iloc[:, 0] - 2.0 * merged.iloc[:, 1] + merged.iloc[:, 2]
                current_fly_level = merged.iloc[-1]['FlyLevel']
                percentile_fly = stats.percentileofscore(merged['FlyLevel'], current_fly_level, kind='mean')
                stdev_fly = np.std(merged['FlyLevel'])
                mean_fly = np.average(merged['FlyLevel'])
                z_fly = (current_fly_level - mean_fly) / stdev_fly
                if legA_RD is not None:
                    merged['RolledDownLevel'] = merged.iloc[:, 3] - 2.0 * merged.iloc[:, 4] + merged.iloc[:, 5]
                    percentile_RD = stats.percentileofscore(merged['RolledDownLevel'], current_fly_level, kind='mean')
                    stdev_RD = np.std(merged['RolledDownLevel'])
                    mean_RD = np.average(merged['RolledDownLevel'])
                    z_RD = (current_fly_level - mean_RD) / stdev_RD
                else:
                    percentile_RD = np.NaN
                    z_RD = np.NaN

                row_dict['Name'] = sym_root
                row_dict['Leg1'] = fly_combos[i][0]
                row_dict['Leg2'] = fly_combos[i][1]
                row_dict['Leg3'] = fly_combos[i][2]
                row_dict['Leg1 Actual'] = get_futures_actual_ticker(meta_data, legA.name)
                row_dict['Leg2 Actual'] = get_futures_actual_ticker(meta_data, legB.name)
                row_dict['Leg3 Actual'] = get_futures_actual_ticker(meta_data, legC.name)
                row_dict['Fly'] = round(current_fly_level, 4)
                row_dict['Fly Prcnt'] = round(percentile_fly, 4)
                row_dict['RD Prcnt'] = round(percentile_RD, 4)
                row_dict['Fly Z-Score'] = round(z_fly, 4)
                row_dict['RD Z-Score'] = round(z_RD, 4)

                df_2 = pd.DataFrame(row_dict, index=[fly_combos[i]])
                df_fly_stats = df_fly_stats.append(df_2)
                logging.info('fly {} {} finished'.format(sym_root, fly_combos[i]))
            except:
                logging.error('fly {} {} failed'.format(sym_root, fly_combos[i]))

        df_fly_stats.to_hdf(os.path.join(global_settings.root_path, 'data/fly_scores.h5'), key=sym_root)

