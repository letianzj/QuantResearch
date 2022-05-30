#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
##################################################
## {Description} Currently
    -- stock from yahoo to stocks
    -- futures from quandl to futures; KC, JO, and MW are missing
    -- USDT from treasury gov to misc
    -- Option Stats from CBOE to misc
    -- VIX Index from CBOE to stocks
    -- VIX futures from CBOE to futures
##################################################
## {License_info}
##################################################
## Author: {Letian Wang}
## Copyright: Copyright {2020}, {Quant Research}
## Credits: [{credit_list}]
## License: {license}
## Version: {mayor}.{minor}.{rel}
## Maintainer: {maintainer}
## Email: {contact_email}
## Status: {dev_status}
##################################################

"""
import os
import io
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import logging
import h5py
import zipfile
from shutil import copyfile
import global_settings
from stocks_downloader import download_stocks_hist_prices, download_stocks_hist_1m_data, download_vix_index_from_cboe
from futures_downloader import download_futures_hist_prices_from_quandl
from misc_downloader import download_treasury_curve_from_gov, download_option_stats_from_cboe, download_current_cot_from_cftc
from curve_constructor import construct_inter_commodity_spreads, construct_comdty_generic_hist_prices, construct_inter_comdty_generic_hist_prices, construct_curve_spread_fly
import data_loader

today = datetime.today()
# os.chdir(global_settings.root_path)
script = os.path.basename(__file__).split('.')[0]

logging.basicConfig(
    level=logging.DEBUG,      # logging.INFO
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"log/{today.strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

def check_h5_file(fname):
    is_valid = True
    if os.path.isfile(fname):
        with h5py.File(fname, 'r') as f:
            for k in f.keys():
                try:
                    _ = pd.read_hdf(fname, key=k)
                except:
                    is_valid = False
                    break
    else:
        is_valid = False

    return is_valid


def main(args):
    logging.info('==' * 110)
    start = time.time()

    if args.stocks:
        try:
            logging.info('-------- download stock prices --------')
            download_stocks_hist_prices()
            logging.info('-------- stock prices updated --------')
        except:
            logging.error('-------- stock prices failed --------')
        time.sleep(3)

        try:
            logging.info('-------- download VIX index --------')
            # download_vix_index_from_cboe()
            logging.info('-------- VIX Index updated --------')
        except:
            logging.error('-------- VIX Index failed --------')
        time.sleep(3)

        try:
            logging.info('-------- download FX rates --------')
            # download_fx_rates_from_ecb()
            logging.info('-------- FX Rates updated --------')
        except:
            logging.error('-------- FX Rates failed --------')
        time.sleep(3)

    if args.intraday:
        try:
            logging.info('-------- download intraday 1m data --------')
            # download_stocks_hist_1m_data()
            logging.info('-------- 1m intraday data succeeded --------')
        except:
            logging.error('-------- 1m intraday data failed --------')
        time.sleep(3)

    if args.futures:
        try:
            logging.info('-------- download futures prices --------')
            download_futures_hist_prices_from_quandl()
            logging.info('-------- futures prices updated --------')
        except:
            logging.error(' --------futures prices failed --------')
        time.sleep(3)

        try:
            logging.info('-------- download VIX futures --------')
            # download_vix_futures_from_cboe()
            logging.info('-------- VIX Futures updated --------')
        except:
            logging.error('-------- VIX futures failed --------')
        time.sleep(3)

    if args.misc:
        # key: PCR:VIX PCR:SPX USDT etc
        misc_dict = data_loader.load_misc()
        try:
            logging.info('-------- download treasury curve --------')
            download_treasury_curve_from_gov(misc_dict)
            logging.info('-------- treasury curve updated --------')
        except:
            logging.error('-------- treasury curve failed --------')
        time.sleep(3)

        try:
            logging.info('-------- download put call ratio --------')
            download_option_stats_from_cboe(misc_dict)
            logging.info('-------- put call ratio updated --------')
        except:
            logging.error('-------- put call ratio failed --------')
        time.sleep(3)

        try:
            logging.info('-------- download COT reports --------')
            download_current_cot_from_cftc(misc_dict)
            logging.info('-------- COT Table updated --------')
        except:
            logging.error('-------- COT Table failed --------')
        time.sleep(3)

        for k in misc_dict.keys():
            misc_dict[k].to_hdf(os.path.join(global_settings.root_path, 'data/misc.h5'), key=k)

    if args.generic:
        try:
            logging.info('-------- Construct generic hist prices --------')
            construct_comdty_generic_hist_prices()
            logging.info('-------- commodity generic prices updated --------')
        except:
            logging.error('-------- commodity generic prices failed --------')
        time.sleep(3)

        try:
            logging.info('-------- Construct ICS --------')
            construct_inter_commodity_spreads()
            logging.info('-------- inter-commodity spread updated --------')
        except:
            logging.error('-------- inter-commodity spread failed --------')
        time.sleep(3)

        try:
            logging.info('-------- Construct ICS generic --------')
            construct_inter_comdty_generic_hist_prices()
            logging.info('inter-commodity generic spread updated.')
        except:
            logging.error('inter-commodity generic spread failed.')
        time.sleep(3)

    if args.curve:
        try:
            logging.info('updating futures spread and fly --------')
            construct_curve_spread_fly()
            logging.info('finished updating futures spread and fly --------')
        except:
            logging.error('futures spread and fly failed.')
        time.sleep(3)        

    # ------------- copy if valid -------------------------- #
    if args.backup:
        logging.info('-------- Backup data h5 --------')
        is_valid = check_h5_file(os.path.join(global_settings.root_path, 'data/misc.h5'))
        if is_valid:
            logging.info('-------- misc backed up --------')
            copyfile(os.path.join(global_settings.root_path, 'data/misc.h5'), os.path.join(global_settings.root_path, 'data/misc_bak.h5'))
        else:
            logging.error('-------- misc corrupted --------')
        is_valid = check_h5_file(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'))
        if is_valid:
            logging.info('-------- futures backed up --------')
            copyfile(os.path.join(global_settings.root_path, 'data/futures_historical_prices.h5'), os.path.join(global_settings.root_path, 'data/futures_historical_prices_bak.h5'))
        else:
            logging.error('-------- futures corrupted --------')
        is_valid = check_h5_file(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5'))
        if is_valid:
            logging.info('-------- stocks backed up --------')
            copyfile(os.path.join(global_settings.root_path, 'data/stocks_historical_prices.h5'), os.path.join(global_settings.root_path, 'data/stocks_historical_prices_bak.h5'))
        else:
            logging.error('-------- stocks corrupted --------')

    end = time.time()
    run_time = round((end - start) / 60.0, 2)
    logging.info('Performance time: {} min'.format(run_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stocks", help="stock downloader", action="store_true")
    parser.add_argument("-i", "--intraday", help="stock intraday downloader", action="store_true")
    parser.add_argument("-f", "--futures", help="futures downloader", action="store_true")
    parser.add_argument("-m", "--misc", help="misc downloader", action="store_true")
    parser.add_argument("-g", "--generic", help="generic constructor", action="store_true")
    parser.add_argument("-c", "--curve", help="curve constructor", action="store_true")
    parser.add_argument("-b", "--backup", help="backup if valid", action="store_true")

    args = parser.parse_args()
    main(args)
