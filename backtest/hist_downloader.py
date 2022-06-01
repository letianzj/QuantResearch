#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import signal
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import yfinance as yf
import pickle
from yahoo_fin import stock_info
from dateutil.parser import parse
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

class TimeoutError(Exception):
    def __init__(self, value = "Timed Out"):
        self.value = value
    def __str__(self):
        return repr(self.value)

# https://stackoverflow.com/questions/35490555/python-timeout-decorator
def timeout(seconds_before_timeout):
    def decorate(f):
        def handler(signum, frame):
            raise TimeoutError()
        def new_f(*args, **kwargs):
            old = signal.signal(signal.SIGALRM, handler)
            old_time_left = signal.alarm(seconds_before_timeout)
            if 0 < old_time_left < seconds_before_timeout: # never lengthen existing timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
            finally:
                if old_time_left > 0: # deduct f's run time from the saved timer
                    old_time_left -= time.time() - start_time
                signal.signal(signal.SIGALRM, old)
                signal.alarm(old_time_left)
            return result
        new_f.func_name = f.func_name
        return new_f
    return decorate

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False

def save(df, fn):
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df.to_csv(fn)

def run(args):
    current_path = os.path.dirname(os.path.abspath(__file__))
    hist_path = os.path.join(current_path, '..', 'data')
    end_date = datetime.today()
    #start_date = end_date + timedelta(days=-5 * 365)
    start_date = datetime(2006, 1, 1)

    if args.sym:
        symbols = args.sym.split('+')
        if 'grp_all' in symbols:
            print('Downloading stock group all .............')
            df_stocks = pd.read_csv(os.path.join(hist_path, 'all_stocks.csv'), header=None)
            df_stocks = df_stocks.iloc[4080:]
            for idx, r in df_stocks.iterrows():
                s = r.iloc[0]
                try:
                    data = yf.download(s, start=start_date, end=end_date)
                    save(data, os.path.join(hist_path, f'{s}.csv'))
                    print(f'{s} is downloaded')
                    time.sleep(3)
                except Exception as e:
                    print(f'{s} failed. {str(e)}')
        elif 'grp_index' in symbols:
            print('Downloading stock group index .............')
            s_dict = {'^GSPC': 'SPX', '^DJI': 'DJI', '^NDX': 'NDX', '^RUT': 'RUT', 'VIX': 'VIX'}
            for k, v in s_dict.items():
                try:
                    data = web.DataReader(name=k, data_source='yahoo', start=start_date, end=end_date)
                    save(data, os.path.join(hist_path, f'{v}.csv'))
                    print(f'{v} is downloaded')
                    time.sleep(3)
                except Exception as e:
                    print(f'{v} failed. {str(e)}')
        elif 'grp_dow' in symbols:
            print('Downloading stock group dow30 .............')
            df_stocks = pd.read_csv(os.path.join(hist_path, 'dow30.csv'), header=None)
            for idx, row in df_stocks.iterrows():
                try:
                    data = web.DataReader(name=row[0], data_source='yahoo', start=start_date, end=end_date)
                    save(data, os.path.join(hist_path, f'{row[0]}.csv'))
                except Exception as e:
                    print(f'{row[0]} failed. {str(e)}')
        elif 'grp_sector' in symbols:
            print('Downloading stock group sector .............')
            df = pd.read_csv(os.path.join(hist_path, 'sectoretf.csv'), header=None)
            for idx, row in df.iterrows():
                try:
                    data = web.DataReader(name=row[0], data_source='yahoo', start=start_date, end=end_date)
                    save(data, os.path.join(hist_path, f'{row[0]}.csv'))
                except Exception as e:
                    print(f'{row[0]} failed. {str(e)}')
            print('Sector ETF downloaded')
        elif 'grp_country' in symbols:
            print('Country ETF downloading .............')
            df = pd.read_csv(os.path.join(hist_path, 'countryetf.csv'), header=None)
            for idx, row in df.iterrows():
                try:
                    data = web.DataReader(name=row[0], data_source='yahoo', start=start_date, end=end_date)
                    save(data, os.path.join(hist_path, f'{row[0]}.csv'))
                except Exception as e:
                    print(f'{row[0]} failed. {str(e)}')
            print('Country ETF downloaded')
        elif 'grp_taa' in symbols:
            print('Mebane Faber TAA downloading .............')
            symbols = ['SPY', 'EFA', 'TIP', 'AGG', 'VNQ', 'GLD', 'GSG']  # sp, em, bond, real estate, gold
            for sym in symbols:
                try:
                    data = web.DataReader(name=sym, data_source='yahoo', start=start_date, end=end_date)
                    save(data, os.path.join(hist_path, f'{sym}.csv'))
                except Exception as e:
                    print(f'{sym} failed. {str(e)}')
            print('Mebane Faber TAA downloaded')
        else:
            for sym in symbols:
                try:
                    data = web.DataReader(name=sym, data_source='yahoo', start=start_date, end=end_date)
                    save(data, os.path.join(hist_path, f'{sym}.csv'))
                except Exception as e:
                    print(f'{sym} failed. {str(e)}')
            print(f'{args.sym} downloaded')

    if args.corp:
        df_stocks = pd.read_csv(os.path.join(hist_path, 'all_stocks.csv'), header=None)
        df_general_info = pd.DataFrame()
        for idx, r in df_stocks.iterrows():
            try:
                s = r.iloc[0]
                ticker = yf.Ticker(s)
                s_interested = {k: v for k, v in ticker.info.items() if k in ['sector', 'industry', 'fullTimeEmployees', 'city', 'state', 'country', 'exchange', 'shortName', 'longName']}
                df_s = pd.DataFrame.from_dict(s_interested, orient='index', columns=[s])
                df_s = df_s.T
                df_general_info = pd.concat([df_general_info, df_s], axis=0)
                print(f'{s} corp info downloaded')
            except Exception as e:
                print(f'{s} corp info download failed, {str(e)}')
            time.sleep(3)
        df_general_info.to_csv(os.path.join(hist_path, 'corporate_info.csv'))

    if args.fundamental:
        call_dict = {'balance_sheet': stock_info.get_balance_sheet,
                     'cash_flow': stock_info.get_cash_flow,
                     'income_statement': stock_info.get_income_statement,
                     'stats_valuation': stock_info.get_stats_valuation,
                     }

        print('Downloading fundamentals .............')
        outfile = os.path.join(hist_path, 'all_stocks.pkl')
        dict_all_stocks = dict()
        if os.path.isfile(outfile):
            with open(outfile, 'rb') as f:
                dict_all_stocks = pickle.load(f)
        df_stocks = pd.read_csv(os.path.join(hist_path, 'all_stocks.csv'), header=None)

        field = args.fundamental
        func_call = call_dict[field]
        for idx, r in df_stocks.iterrows():
            s = r.iloc[0]
            if s not in dict_all_stocks.keys():
                dict_all_stocks[s] = dict()

            if field in dict_all_stocks[s].keys():
                df_old = dict_all_stocks[s][field]
            else:
                df_old = pd.DataFrame()
            if not isinstance(df_old, pd.DataFrame):
                df_old = pd.DataFrame()
            try:
                df_new = func_call(s)
                df_new.set_index(df_new.columns[0], inplace=True)
                df_new.index.name = 'Breakdown'
                cols = [c for c in df_new.columns if is_date(c)]
                df_new = df_new[cols]
                # combine_first is convenient
                df_new = df_new.combine_first(df_old)
                dict_all_stocks[s][field] = df_new
                print(f'{s} {field} is downloaded, having {df_new.shape} records')
                time.sleep(3)
            except Exception as e:
                print(f'{s} {field} failed; {str(e)}')

        with open(outfile, 'wb') as f:
            pickle.dump(dict_all_stocks, f, pickle.HIGHEST_PROTOCOL)
        print(f'Fundamentals {field} downloaded')

    # This is adapted from https://towardsdatascience.com/sentiment-analysis-of-stocks-from-financial-news-using-python-82ebdcefb638
    if args.sentiment:
        print('sentiment downloading .............')
        finwiz_url = 'https://finviz.com/quote.ashx?t='

        outfile = os.path.join(hist_path, 'all_stocks.pkl')
        dict_all_stocks = dict()
        if os.path.isfile(outfile):
            with open(outfile, 'rb') as f:
                dict_all_stocks = pickle.load(f)
        df_stocks = pd.read_csv(os.path.join(hist_path, 'intraday_stocks.csv'), header=None)
        field = 'sentiment'
        for idx, r in df_stocks.iterrows():
            s = r.iloc[0]
            if s not in dict_all_stocks.keys():
                dict_all_stocks[s] = dict()

            if field in dict_all_stocks[s].keys():
                list_old = dict_all_stocks[s][field]
            else:
                list_old = []
            if not isinstance(list_old, list):
                list_old = []
            try:
                url = finwiz_url + s
                req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
                response = urlopen(req)
                # Read the contents of the file into 'html'
                html = BeautifulSoup(response)
                # Find 'news-table' in the Soup and load it into 'news_table'
                news_table = html.find(id='news-table')
                parsed_news = []

                # Iterate through all tr tags in 'news_table'
                insert_idx = 0
                for x in news_table.findAll('tr'):
                    # read the text from each tr tag into text
                    # get text from a only
                    text = x.a.get_text()
                    # split text in the td tag into a list
                    date_scrape = x.td.text.split()
                    # if the length of 'date_scrape' is 1, load 'time' as the only element
                    if len(date_scrape) == 1:
                        tm = date_scrape[0]
                    # else load 'date' as the 1st element and 'time' as the second
                    else:
                        dt = date_scrape[0]
                        tm = date_scrape[1]

                    if [s, dt, tm, text] not in list_old:
                        print(f'insert {s} {dt} {tm} at {insert_idx}')
                        list_old.insert(insert_idx, [s, dt, tm, text])
                        insert_idx += 1
                    else:
                        print(f'skip {s} {dt} {tm}')

                dict_all_stocks[s][field] = list_old
                print(f'{s} {field} is downloaded')
                time.sleep(3)
            except Exception as e:
                print(f'{s} {field} failed; {str(e)}')

        with open(outfile, 'wb') as f:
            pickle.dump(dict_all_stocks, f, pickle.HIGHEST_PROTOCOL)

        print('sentiment downloaded')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Historical Downloader')
    parser.add_argument('--sym', help='AAPL+AMZN or grp_all, grp_dow, grp_sector, grp_country, grp_taa')
    parser.add_argument('--corp', action='store_true', help='corporate info')
    parser.add_argument('--fundamental', help='balance_sheet cash_flow income_statement stats_valuation')
    parser.add_argument('--sentiment', action='store_true')

    args = parser.parse_args()
    run(args)