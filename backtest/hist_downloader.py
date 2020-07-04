#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pandas_datareader.data as web
# import yfinance as yf

def save(df, fn):
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df.to_csv(fn)

def run(args):
    current_path = os.path.dirname(os.path.abspath(__file__))
    hist_path = os.path.join(current_path, '..\data')
    end_date = datetime.today()
    #start_date = end_date + timedelta(days=-5 * 365)
    start_date = datetime(2006, 1, 1)

    if args.index:
        print('Index downloading .............')
        data = web.DataReader(name='^GSPC', data_source='yahoo', start=start_date, end=end_date)
        save(data, os.path.join(hist_path, f'SPX.csv'))
        data = web.DataReader(name='^DJI', data_source='yahoo', start=start_date, end=end_date)
        save(data, os.path.join(hist_path, f'DJI.csv'))
        data = web.DataReader(name='^NDX', data_source='yahoo', start=start_date, end=end_date)
        save(data, os.path.join(hist_path, f'NDX.csv'))
        data = web.DataReader(name='^RUT', data_source='yahoo', start=start_date, end=end_date)
        save(data, os.path.join(hist_path, f'RUT.csv'))
        print('Index downloaded')

    if args.dow:
        print('Dow30 downloading .............')
        df = pd.read_csv(os.path.join(hist_path, 'dow30.csv'), header=None)
        for idx, row in df.iterrows():
            try:
                data = web.DataReader(name=row[0], data_source='yahoo', start=start_date, end=end_date)
                save(data, os.path.join(hist_path, f'{row[0]}.csv'))
            except Exception as e:
                print(f'{row[0]} failed. {str(e)}')
        print('Dow30 downloaded')

    if args.sector:
        print('Sector ETF downloading .............')
        df = pd.read_csv(os.path.join(hist_path, 'sectoretf.csv'), header=None)
        for idx, row in df.iterrows():
            try:
                data = web.DataReader(name=row[0], data_source='yahoo', start=start_date, end=end_date)
                save(data, os.path.join(hist_path, f'{row[0]}.csv'))
            except Exception as e:
                print(f'{row[0]} failed. {str(e)}')
        print('Sector ETF downloaded')

    if args.country:
        print('Country ETF downloading .............')
        df = pd.read_csv(os.path.join(hist_path, 'countryetf.csv'), header=None)
        for idx, row in df.iterrows():
            try:
                data = web.DataReader(name=row[0], data_source='yahoo', start=start_date, end=end_date)
                save(data, os.path.join(hist_path, f'{row[0]}.csv'))
            except Exception as e:
                print(f'{row[0]} failed. {str(e)}')
        print('Country ETF downloaded')

    if args.taa:
        print('Mebane Faber TAA downloading .............')
        symbols = ['SPY', 'EFA', 'AGG', 'VNQ', 'GLD']   # sp, em, bond, real estate, gold
        for sym in symbols:
            try:
                data = web.DataReader(name=sym, data_source='yahoo', start=start_date, end=end_date)
                save(data, os.path.join(hist_path, f'{sym}.csv'))
            except Exception as e:
                print(f'{sym} failed. {str(e)}')
        print('Mebane Faber TAA downloaded')

    if args.sym:
        print(f'{args.sym} downloading .............')
        symbols = args.sym.split('+')
        for sym in symbols:
            try:
                data = web.DataReader(name=sym, data_source='yahoo', start=start_date, end=end_date)
                save(data, os.path.join(hist_path, f'{sym}.csv'))
            except Exception as e:
                print(f'{sym} failed. {str(e)}')
        print(f'{args.sym} downloaded')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Historical Downloader')
    parser.add_argument('--index',  action='store_true')
    parser.add_argument('--dow',  action='store_true')
    parser.add_argument('--sector',  action='store_true')
    parser.add_argument('--country',  action='store_true')
    parser.add_argument('--taa', action='store_true')
    parser.add_argument('--sym', help='symbol')

    args = parser.parse_args()
    run(args)