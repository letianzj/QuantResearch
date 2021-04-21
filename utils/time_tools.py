#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta


def convert_date_input(input_str, default_date=None):
    """
    convert date input
    :param input_str: 3y, 2m, 0mm 1w, or yyyy-mm-dd
    :param default_date: datetime.date
    :return:datetime.date
    """
    ret_date = datetime.today()

    try:
        if 'Y' in input_str or 'y' in input_str:
            yr = int(input_str[:-1])
            ret_date = ret_date.replace(year=ret_date.year+yr)
        elif 'M' in input_str or 'm' in input_str:
            mth = int(input_str[:-1])
            total_mth = ret_date.month + mth
            nm = total_mth % 12
            ny = int((total_mth - nm) / 12)
            ret_date = ret_date.replace(year=ret_date.year+ny)
            ret_date = ret_date.replace(month=nm)
        elif 'W' in input_str or 'w' in input_str:
            wks = int(input_str[:-1])
            ret_date = ret_date + timedelta(days=7*wks)
        elif 'D' in input_str or 'd' in input_str:
            ds = int(input_str[:-1])
            ret_date = ret_date + timedelta(days=ds)
        else:
            ret_date = datetime.strptime(input_str, '%Y-%m-%d')
    except:
        # ret_date = ret_date + timedelta(days=-5 * 365)
        ret_date = default_date
    return ret_date


def locate_week():
    today = datetime.today()
    return [today + timedelta(days=i) for i in range(0 - today.weekday(), 7 - today.weekday())]    # week of today, then intersect with datetimeindex

