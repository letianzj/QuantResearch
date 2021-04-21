#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import List, Set, Dict, Tuple, Sequence, Optional
import yfinance as yf
from bs4 import BeautifulSoup
import requests
import csv
import h5py
import quandl
import logging
import global_settings

def download_treasury_curve_from_quandl(misc_dict: Dict) -> None:
    """
        download treasury curve and save to pickle
        :return:
        """
    start_date = datetime(2010, 1, 1)
    end_date = datetime.today()
    start_date = end_date + timedelta(days=-global_settings.lookback_days)

    df = quandl.get('USTREASURY/YIELD', start_date=start_date, end_date=end_date,
                    authtoken=global_settings.quandl_auth)

    misc_dict['USDT'] = df.combine_first(misc_dict['USDT'])


def download_treasury_curve_from_gov(misc_dict: Dict) -> None:
    # rates_old = pd.read_csv(os.path.join(global_settings.root_path, 'data/usd_rates.csv'), index_col=0, header=0)
    url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield'
    page = requests.get(url)
    # print(page.status_code)   # should be 200
    soup = BeautifulSoup(page.content, 'html.parser')
    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))
    df = df[1]
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    misc_dict['USDT'] = df.combine_first(misc_dict['USDT'])
    misc_dict['USDT'].sort_index(inplace=True)


def download_vix_from_quandl(misc_dict: Dict) -> None:
    """
        download vix curve and save to pickle
        :return:
        """
    start_date = datetime(2010, 1, 1)
    end_date = datetime.today()
    start_date = end_date + timedelta(days=-global_settings.lookback_days)

    for i in range(1, 10):
        df = quandl.get('CHRIS/CBOE_VX'+str(i), start_date=start_date, end_date=end_date,
                        authtoken=global_settings.quandl_auth)
    # for c in misc_dict['PCR:VIX'].columns:
    #         misc_dict['PCR:VIX'][c] = misc_dict['PCR:VIX'][c].astype(np.int64)


# https://markets.cboe.com/us/options/market_statistics/daily/
def download_option_stats_from_cboe(misc_dict: Dict) -> None:
    end_date = datetime.today()
    date_list = [end_date - timedelta(days=x) for x in range(global_settings.lookback_days)]
    symbols = ['CBOE VOLATILITY INDEX (VIX)', 'SPX + SPXW', 'INDEX OPTIONS']
    symbols_not = ['CBOE VOLATILITY INDEX (VIX) PUT/CALL RATIO', 'SPX + SPXW PUT/CALL RATIO', 'INDEX PUT/CALL RATIO']

    for asofdate in date_list:
        dtstr = asofdate.strftime('%Y-%m-%d')
        url = f'https://markets.cboe.com/us/options/market_statistics/daily/?mkt=cone&dt={dtstr}'
        webpage = requests.get(url)
        soup = BeautifulSoup(webpage.content, 'html.parser')
        tables = soup.find_all('table')
        rmt = re.search('\d{4}-\d{2}-\d{2}', webpage.content.decode())
        target_date = datetime.strptime(rmt.group(), '%Y-%m-%d')
        for sym_idx in range(3):
            symbol = symbols[sym_idx]
            symbol_not = symbols_not[sym_idx]
            for tbl in tables:
                try:
                    if symbol in str(tbl) and symbol_not not in str(tbl):
                        df = pd.read_html(str(tbl))
                        df = df[0]
                        df.columns = df.iloc[0]
                        df.set_index(df.columns[0], inplace=True)

                        row_dict = {}
                        row_dict['CV'] = np.int64(df.loc['VOLUME', 'CALL'])
                        row_dict['PV'] = np.int64(df.loc['VOLUME', 'PUT'])
                        row_dict['COI'] = np.int64(df.loc['OPEN INTEREST', 'CALL'])
                        row_dict['POI'] = np.int64(df.loc['OPEN INTEREST', 'PUT'])
                        df_row = pd.DataFrame(row_dict, index=[target_date.date()])

                        if sym_idx == 0:
                            if 'PCR:SPX' in misc_dict.keys():
                                misc_dict['PCR:VIX'] = df_row.combine_first(misc_dict['PCR:VIX'])
                            else:
                                misc_dict['PCR:VIX'] = df_row
                        elif sym_idx == 1:
                            if 'PCR:SPX' in misc_dict.keys():
                                misc_dict['PCR:SPX'] = df_row.combine_first(misc_dict['PCR:SPX'])
                            else:
                                misc_dict['PCR:SPX'] = df_row
                        elif sym_idx == 2:
                            if 'PCR:INDEX' in misc_dict.keys():
                                misc_dict['PCR:INDEX'] = df_row.combine_first(misc_dict['PCR:INDEX'])
                            else:
                                misc_dict['PCR:INDEX'] = df_row
                except:
                    continue
            time.sleep(1)
        logging.info(asofdate.strftime("%Y-%m-%d") + ' is done')


def download_current_cot_from_cftc(misc_dict: Dict) -> None:
    df_futures_meta = pd.read_csv(os.path.join(global_settings.root_path, 'data/config/futures_meta.csv'), keep_default_na =True, index_col=0)
    df_futures_meta.dropna(subset=['COT_CODE'], axis=0, inplace=True)
    cot_dict = {}
    cot_c_dict = {}

    # Disaggregated, futures_only
    url_f = 'https://www.cftc.gov/dea/newcot/f_disagg.txt'
    r_f = requests.get(url_f, stream=True)
    if r_f.ok:
        data_f = r_f.content.decode('utf8')
    with open(r'C:\Users\letian\Downloads\f_year.txt') as csv_file:
        # for row in csv.reader(data_f.split('\n'), delimiter=',', quotechar='"'):
        for row in csv.reader(csv_file, delimiter=',', quotechar='"'):
            if row == []:
                continue

            contract = df_futures_meta.loc[df_futures_meta['COT_CODE'] == row[3]]
            if contract.shape[0] != 1:
                continue

            row_dict = {}
            sym = contract.index[0]
            d = row[2].split("-")
            report_date = datetime(int(d[0]), int(d[1]), int(d[2]))
            row_dict['Open Interest:F'] = int(row[7])
            row_dict['Producer/Merchant/Processor/User:Long:F'] = int(row[8])
            row_dict['Producer/Merchant/Processor/User:Short:F'] = int(row[9])
            row_dict['Swap Dealers:Long:F'] = int(row[10])
            row_dict['Swap Dealers:Short:F'] = int(row[11])
            row_dict['Swap Dealers:Spreading:F'] = int(row[12])
            row_dict['Managed Money:Long:F'] = int(row[13])
            row_dict['Managed Money:Short:F'] = int(row[14])
            row_dict['Managed Money:Spreading:F'] = int(row[15])
            row_dict['Other Reportables:Long:F'] = int(row[16])
            row_dict['Other Reportables:Short:F'] = int(row[17])
            row_dict['Other Reportables:Spreading:F'] = int(row[18])
            row_dict['Nonreportable Positions:Long:F'] = int(row[21])
            row_dict['Nonreportable Positions:Short:F'] = int(row[22])

            df_row = pd.DataFrame(row_dict, index=[report_date], columns=row_dict.keys())

            if f'COT:{sym}' not in cot_dict.keys():
                cot_dict[f'COT:{sym}'] = df_row.copy()
            else:
                cot_dict[f'COT:{sym}'] = pd.concat([cot_dict[f'COT:{sym}'], df_row], axis=0)

            logging.info(f'COT:{sym} futures is downloaded')

    # Disaggregated, futures and options
    url_c = 'https://www.cftc.gov/dea/newcot/c_disagg.txt'
    r_c = requests.get(url_c, stream=True)
    if r_c.ok:
        data_c = r_c.content.decode('utf8')
    with open(r'C:\Users\letian\Downloads\c_year.txt') as csv_file:           # csv
        # for row in csv.reader(data_c.split('\n'), delimiter=',', quotechar='"'):
        for row in csv.reader(csv_file, delimiter=',', quotechar='"'):       # csv
            if row == []:
                continue

            contract = df_futures_meta.loc[df_futures_meta['COT_CODE'] == row[3]]
            if contract.shape[0] != 1:
                continue

            row_dict = {}
            sym = contract.index[0]
            d = row[2].split("-")
            report_date = datetime(int(d[0]), int(d[1]), int(d[2]))
            row_dict['Open Interest:C'] = int(row[7])
            row_dict['Producer/Merchant/Processor/User:Long:C'] = int(row[8])
            row_dict['Producer/Merchant/Processor/User:Short:C'] = int(row[9])
            row_dict['Swap Dealers:Long:C'] = int(row[10])
            row_dict['Swap Dealers:Short:C'] = int(row[11])
            row_dict['Swap Dealers:Spreading:C'] = int(row[12])
            row_dict['Managed Money:Long:C'] = int(row[13])
            row_dict['Managed Money:Short:C'] = int(row[14])
            row_dict['Managed Money:Spreading:C'] = int(row[15])
            row_dict['Other Reportables:Long:C'] = int(row[16])
            row_dict['Other Reportables:Short:C'] = int(row[17])
            row_dict['Other Reportables:Spreading:C'] = int(row[18])
            row_dict['Nonreportable Positions:Long:C'] = int(row[21])
            row_dict['Nonreportable Positions:Short:C'] = int(row[22])

            df_row = pd.DataFrame(row_dict, index=[report_date], columns=row_dict.keys())
            if f'COT:{sym}' not in cot_c_dict.keys():
                cot_c_dict[f'COT:{sym}'] = df_row.copy()
            else:
                cot_c_dict[f'COT:{sym}'] = pd.concat([cot_c_dict[f'COT:{sym}'], df_row], axis=0)

            logging.info(f'COT:{sym} futures and options is downloaded')

    for key, value in cot_dict.items():
        df = pd.concat([value, cot_c_dict[key]], axis=1)
        if key in misc_dict.keys():
            misc_dict[key] = df.combine_first(misc_dict[key])
        else:
            misc_dict[key] = df
        misc_dict[key].sort_index(inplace=True)

    cot_dict = {}
    cot_c_dict = {}
    # financial, futures_only
    url_f = 'https://www.cftc.gov/dea/newcot/FinFutWk.txt'
    r_f = requests.get(url_f, stream=True)
    if r_f.ok:
        data_f = r_f.content.decode('utf8')
    with open(r'C:\Users\letian\Downloads\FinFutYY.txt') as csv_file:
        # for row in csv.reader(data_f.split('\n'), delimiter=',', quotechar='"'):
        for row in csv.reader(csv_file, delimiter=',', quotechar='"'):
            if row == []:
                continue

            contract = df_futures_meta.loc[df_futures_meta['COT_CODE'] == row[3]]
            if contract.shape[0] != 1:
                continue

            row_dict = {}
            sym = contract.index[0]
            d = row[2].split("-")
            report_date = datetime(int(d[0]), int(d[1]), int(d[2]))
            row_dict['Open Interest:F'] = int(row[7])
            row_dict['Dealer Intermediary:Long:F'] = int(row[8])
            row_dict['Dealer Intermediary:Short:F'] = int(row[9])
            row_dict['Dealer Intermediary:Spreading:F'] = int(row[10])
            row_dict['Asset Manager/Institutional:Long:F'] = int(row[11])
            row_dict['Asset Manager/Institutional:Short:F'] = int(row[12])
            row_dict['Asset Manager/Institutional:Spreading:F'] = int(row[13])
            row_dict['Leveraged Funds:Long:F'] = int(row[14])
            row_dict['Leveraged Funds:Short:F'] = int(row[15])
            row_dict['Leveraged Funds:Spreading:F'] = int(row[16])
            row_dict['Other Reportables:Long:F'] = int(row[17])
            row_dict['Other Reportables:Short:F'] = int(row[18])
            row_dict['Other Reportables:Spreading:F'] = int(row[19])
            row_dict['Nonreportable Positions:Long:F'] = int(row[22])
            row_dict['Nonreportable Positions:Short:F'] = int(row[23])

            df_row = pd.DataFrame(row_dict, index=[report_date], columns=row_dict.keys())
            if f'COT:{sym}' not in cot_dict.keys():
                cot_dict[f'COT:{sym}'] = df_row.copy()
            else:
                cot_dict[f'COT:{sym}'] = pd.concat([cot_dict[f'COT:{sym}'], df_row], axis=0)

            logging.info(f'COT:{sym} futures is downloaded')

    # financial, futures and options
    url_c = 'https://www.cftc.gov/dea/newcot/FinComWk.txt'
    r_c = requests.get(url_c, stream=True)
    if r_c.ok:
        data_c = r_c.content.decode('utf8')
    with open(r'C:\Users\letian\Downloads\FinComYY.txt') as csv_file:
        # for row in csv.reader(data_c.split('\n'), delimiter=',', quotechar='"'):
        for row in csv.reader(csv_file, delimiter=',', quotechar='"'):
            if row == []:
                continue

            contract = df_futures_meta.loc[df_futures_meta['COT_CODE'] == row[3]]
            if contract.shape[0] != 1:
                continue

            row_dict = {}
            sym = contract.index[0]
            d = row[2].split("-")
            report_date = datetime(int(d[0]), int(d[1]), int(d[2]))
            row_dict['Open Interest:C'] = int(row[7])
            row_dict['Dealer Intermediary:Long:C'] = int(row[8])
            row_dict['Dealer Intermediary:Short:C'] = int(row[9])
            row_dict['Dealer Intermediary:Spreading:C'] = int(row[10])
            row_dict['Asset Manager/Institutional:Long:C'] = int(row[11])
            row_dict['Asset Manager/Institutional:Short:C'] = int(row[12])
            row_dict['Asset Manager/Institutional:Spreading:C'] = int(row[13])
            row_dict['Leveraged Funds:Long:C'] = int(row[14])
            row_dict['Leveraged Funds:Short:C'] = int(row[15])
            row_dict['Leveraged Funds:Spreading:C'] = int(row[16])
            row_dict['Other Reportables:Long:C'] = int(row[17])
            row_dict['Other Reportables:Short:C'] = int(row[18])
            row_dict['Other Reportables:Spreading:C'] = int(row[19])
            row_dict['Nonreportable Positions:Long:C'] = int(row[22])
            row_dict['Nonreportable Positions:Short:C'] = int(row[23])

            df_row = pd.DataFrame(row_dict, index=[report_date], columns=row_dict.keys())
            if f'COT:{sym}' not in cot_c_dict.keys():
                cot_c_dict[f'COT:{sym}'] = df_row.copy()
            else:
                cot_c_dict[f'COT:{sym}'] = pd.concat([cot_c_dict[f'COT:{sym}'], df_row], axis=0)

            logging.info(f'COT:{sym} futures and options is downloaded')

    # back to misc.dict
    for key, value in cot_dict.items():
        df = pd.concat([value, cot_c_dict[key]], axis=1)
        if key in misc_dict.keys():
            misc_dict[key] = df.combine_first(misc_dict[key])
        else:
            misc_dict[key] = df
        misc_dict[key].sort_index(inplace=True)
