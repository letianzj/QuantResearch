#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List
import logging


def get_futures_chain(meta_data: pd.DataFrame, asofdate: datetime.date) -> pd.DataFrame:
    """
    get current futures chain on asofdate
    :param meta_data: dataframe: actual futures ==> last trading day
    :param asofdate: datetime.date
    :return: dataframe non-expired actual futures ==> last trading day
    """
    # searchsorted searches the first one >= the given;
    # if the given is the biggest, return last_idx+1 (consistent with range)
    # if the given is the smallest, return 0 (NOT -1)
    dateidx = meta_data['Last_Trade_Date'].searchsorted(asofdate)
    return meta_data.iloc[dateidx:]


def get_futures_generic_ticker(futures_chain: pd.DataFrame, futures: str) -> str:
    """
    get the generic ticker, e.g., NGZ19 ==> NG1
    :param futures_chain, pd.DataFrame NGZ19 indexed: futures_chain on asofdate
    :param futures: actual futures
    :return: generic futures
    """
    contract_idx = list(futures_chain.index).index(futures)
    return futures[:-3] + str(contract_idx+1)


def get_futures_actual_ticker(futures_chain: pd.DataFrame, generic_ticker: str) -> str:
    """
    get the actual ticker, e.g., NG1 ==> NGZ19
    :param futures_chain: futures_chain on asofdate
    :param generic_ticker: generic ticker
    :return: actual ticker
    """
    if generic_ticker[-2].isdigit():
        contract_idx = int(generic_ticker[-2:]) - 1
    else:
        contract_idx = int(generic_ticker[-1]) - 1
    return futures_chain.index[contract_idx]


def get_generic_futures_hist_data(actual_futures_hist_data: pd.DataFrame, meta_data: pd.DataFrame) -> pd.DataFrame:
    """
    .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .
    .   .   .   .   .   .   .   .          <- roll_idx_old
        .   .   .   .   .   .   .          <- roll_idx_previous = roll_idx_old + 1
        .   .   .   .   .   .   .
        .   .   .   .   .   .   .          <- roll_idx_new
            .   .   .   .   .   .
            .   .   .   .   .   .
            .   .   .   .   .   .          <- asofdate/dateidx

    construct generic futures hist data from actual futures hist data
    It assume roll on last trading day, to stitch together PRICE series
    iteratively between roll_idx_previous and roll_idx
    :param actual_futures_hist_data: dataframe, index is date, column is futures
    :param meta_data: dataframe: actual futures ==> last_trading_day
    :return: dataframe: index is date, column is generic futures
    """
    if ':' not in actual_futures_hist_data.columns[0]:
        root_sym = actual_futures_hist_data.columns[0][:-5]
    else:       # inter-comdty spread
        root_sym_idx = actual_futures_hist_data.columns[0].rfind(':')
        root_sym = actual_futures_hist_data.columns[0][:(root_sym_idx+1)]
    asofdate = actual_futures_hist_data.index[-1]
    dateidx = meta_data['Last_Trade_Date'].searchsorted(asofdate)          # first non-expired contract
    n_contracts = min(60, meta_data[dateidx:].shape[0])
    generic_data_df = pd.DataFrame()
    roll_idx_previous = 0

    try:
        for idx in range(dateidx+1):           # contracts expired up to first non-expired included
            # cut at this point, between here and previous cut, they are the 60 generic contracts for this date range
            roll_idx = actual_futures_hist_data.index.searchsorted(meta_data['Last_Trade_Date'].iloc[idx])       # first expired contract, last trade date
            try:
                # from to roll_idx+1; range needs to be idx+1
                actual_contracts = meta_data.index[idx:n_contracts+idx]
                actual_contracts_existed = set(actual_contracts).intersection(set(actual_futures_hist_data.columns))
                actual_contracts_non_existed = set(actual_contracts).difference(set(actual_futures_hist_data.columns))

                if len(actual_contracts_existed) == 0:         # no contract existed
                    continue
                else:
                    temp_df = actual_futures_hist_data.iloc[roll_idx_previous:roll_idx+1][list(actual_contracts_existed)]
                    if len(actual_contracts_non_existed) > 0:
                        empty_temp_df = pd.DataFrame(np.nan, temp_df.index,  columns=list(actual_contracts_non_existed))
                        temp_df = pd.concat([temp_df, empty_temp_df], axis=1, join='outer', sort=True)

                    temp_df = temp_df[list(actual_contracts)]
                    temp_df.columns = [root_sym+str(c+1) for c in range(n_contracts)]
                generic_data_df = generic_data_df.append(temp_df)
            except:
                logging.error(root_sym + ' generic error')
            roll_idx_previous = roll_idx+1
    except:
        pass

    return generic_data_df


def get_seasonal_contracts(futures_asofdate: pd.Timestamp, contracts: List[str], weights: List[int], hist_data: pd.DataFrame, meta_data: pd.DataFrame) -> pd.DataFrame:
    """
    return seasonal series
    :param hist_data:
    :param meta_data:
    :param contracts: outright, curve, fly, e.g.['CLH2021', 'CLM2021'] asof 12/1/2019
    :param weights: matches contracts, e.g. [-1, 1]
    :return: dataframe
    """
    # go back year by year until first leg expires, e.g. on 2/20/2019 CLH2019 expired
    # then find the total business days to its expiry
    # (anchor_day, anchor_contract) pair is (asofdate, first leg contract) pair going back yrs_back
    # e.g. (12/1/2018, CLH2020), (12/1/2017, CLH2019), ...., 
    # the first one is not complete/ not yet expired as of 12/1/2019, while the second one is complete/expired.
    yrs_back = 0
    anchor_days = []
    anchor_contracts = []
    anchor_days.append(futures_asofdate)             # 12/1/2019
    anchor_contracts.append(contracts[0])            # CLH2021
    last_complete_yr = None
    while True:
        try:
            anchor_contract = str(int(anchor_contracts[-1][-2:]) - 1)
            if len(anchor_contract) == 1:
                anchor_contract = '0' + anchor_contract           # CLH10 ==> CLH09  padding 0
            anchor_contract = anchor_contracts[-1][:-2] + anchor_contract
            anchor_day = anchor_days[-1]
            anchor_day = anchor_day.replace(year=anchor_day.year - 1)
            anchor_day = hist_data.index[hist_data.index.searchsorted(anchor_day)]
            if anchor_day <= hist_data.index[0]:  # run out of hist data
                break
            anchor_days.append(anchor_day)                     # 12/1/2019, add 12/1/2018, 12/1/2017 (complete), ...
            anchor_contracts.append(anchor_contract)           # CLH2021, add CLH2020, CLH2019 (complete), ...

            yrs_back -= 1
            if meta_data.loc[anchor_contracts[-1], 'Last_Trade_Date'] < futures_asofdate and last_complete_yr is None:
                last_complete_yr = yrs_back  # in this case, -2
        except:
            break

    s = pd.DataFrame()
    final_index = None
    for i in range(len(anchor_days)):
        # for i in range(len(anchor_days)-1, -1, -1):
        anchor_day = anchor_days[i]
        c1 = str(int(contracts[0][-2:]) - i)
        if len(c1) == 1:
            c1 = '0' + c1
        c1 = contracts[0][:-2] + c1
        s1 = hist_data[c1]

        if (len(contracts) > 1):
            c2 = str(int(contracts[1][-2:]) - i)
            if len(c2) == 1:
                c2 = '0' + c2
            c2 = contracts[1][:-2] + c2
            s2 = hist_data[c2]
            if (len(contracts) > 2):
                c3 = str(int(contracts[2][-2:]) - i)
                if len(c3) == 1:
                    c3 = '0' + c3
                c3 = contracts[2][:-2] + c3
                s3 = hist_data[c3]
                combo = s1 * weights[0] + s2 * weights[1] + s3 * weights[2]
                combo.name = c1 + '-' + c2 + '-' + c3
            else:
                combo = s1 * weights[0] + s2 * weights[1]
                combo.name = c1 + '-' + c2
        else:
            combo = s1 * weights[0]
            combo.name = c1

        j = abs(last_complete_yr)         # j=2
        if i < j:         # when i=0, (12/1/2019, CLH2021) is not complete or expired, i=1, (12/1/2018, CLH2020) is not complete or expired ==> need to append nan
            anchor_day_j = anchor_days[j - i]            # when i=0, days_to_go between (12/1/2019, CLH21)==(12/1/2017, CLH19), when i=1, days_to_go betwen (12/1/2018, CLH20)==(12/1/2017, CLH19)
            anchor_contract_j = anchor_contracts[j]
            days_to_go = hist_data.index.searchsorted(meta_data.loc[anchor_contract_j, 'Last_Trade_Date']) \
                         - hist_data.index.searchsorted(anchor_day_j)    # last_trade_date(CLH19) - 12/1/2017
            # append nan to days_to_go, asofdate should be day days_to_go, there are 0..(days_to_go-1) days with NaN to complete
            s_to_go = pd.Series(np.zeros(days_to_go) + np.nan)
            s_to_go.name = combo.name
            combo.index = range(combo.shape[0] - 1 + days_to_go, days_to_go - 1, -1)  # n-1, n-2, ..., 0
            combo = combo.append(s_to_go)
            combo.sort_index(inplace=True)        # index by days_to_go, or last day first, to facilitate slicing
        else:           # when i=2, (12/1/2017, CLH19) is complete
            last_day_1 = meta_data.loc[c1, 'Last_Trade_Date']
            # anchor_day_idx = hist_data.index.searchsorted(anchor_day)
            # contract_day_idx = hist_data.index.searchsorted(last_day_1)
            combo = combo.loc[:last_day_1]
            if (i == j):
                final_index = combo.index          # the index for all seasonal series
            combo.index = range(combo.shape[0] - 1, -1, -1)  # n-1, n-2, ..., 0

        s = pd.concat([s, combo], axis=1)

    s = s[:len(final_index)]      # cut off s in order to attach final_index
    s.index = final_index.sort_values(ascending=False)   # reverse final_index and attach
    s.sort_index(inplace=True)
    s.index = s.index + pd.offsets.DateOffset(years=1)   # add one year to today

    return s

