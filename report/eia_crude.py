import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil.parser import parse

# --------------------------------------------- Create contents ----------------------------------------------------- #
# 10:30 a.m. (Eastern Time) on Wednesday. Delayed by one day if holiday.
# https://www.eia.gov/petroleum/data.php
def generate_html(today):
    url = 'https://www.eia.gov/petroleum/supply/weekly/'
    page = requests.get(url)
    content = page.content.decode('utf-8')
    idx = content.find("Release Date")
    result = parse(content[idx:idx+100], fuzzy_with_tokens=True)
    release_date = result[0]

    if release_date.date() != today.date():
        return None

    data = pd.read_excel('http://ir.eia.gov/wpsr/psw01.xls', 'Data 1', header=1, skiprows=range(2, 3))
    data.rename(columns={'Sourcekey': 'Date'}, inplace=True)
    data.set_index('Date', inplace=True)
    data = data[['WCESTUS1', 'WGTSTUS1', 'WDISTUS1', 'WPRSTUS1', 'WCSSTUS1', 'WTTSTUS1']]
    data.columns = ['Crude', 'Gasoline', 'Distillate', 'Propane and Other', 'Crude SPR', 'Total']  # Other has place holder
    data = data / 1000.0
    data['Propane and Other'] = data['Total'] - data['Crude'] - data['Gasoline'] - data['Distillate'] - data['Crude SPR']
    df_stock = pd.DataFrame(index=data.columns)
    df_stock['Stock'] = data.iloc[-1]
    df_stock['LastWeek'] = data.iloc[-1] - data.iloc[-2]
    df_stock['LastYear'] = data.iloc[-1] - data.iloc[-52:].mean()
    df_stock['Last5Year'] = data.iloc[-1] - data.iloc[-52*5:].mean()

    # model driven estimate rather than a survey
    # while stock change is from survey, the difference is adjustment.
    # Domestic Production + Net Imports - To Refineries + Adjustment = Stock Change
    data2 = pd.read_excel('http://ir.eia.gov/wpsr/psw01.xls', 'Data 2', header=1, skiprows=range(2, 3))
    data2.rename(columns={'Sourcekey': 'Date'}, inplace=True)
    data2.set_index('Date', inplace=True)
    data2 = data2[
        ['WCRFPUS2', 'W_EPC0_FPF_SAK_MBBLD', 'W_EPC0_FPF_R48_MBBLD', 'WCRNTUS2', 'WCRIMUS2', 'WCEIMUS2', 'WCSIMUS2',
         'W_EPC0_IMU_NUS-Z00_MBBLD', 'WCREXUS2', 'W_EPC0_SCG_NUS_MBBLD', 'WCESCUS2', 'WCSSCUS2', 'WCRAUUS2',
         'WCRRIUS2']]
    data2.columns = ['Domestic Production', '...Alaska', '...Lower 48', 'Net Imports Including SPR', '...Imports',
                    '......Commercial Crude Oil', '......Imoprts by SPR', '......Imoprts into SPR by Others',
                    '...Exports', 'Stock Change', '...Commercial Stock Change', '...SPR Stock Change', 'Adjustment',
                    'Crude Oil Input to Refineries']
    data2 = data2 / 1000.0
    df_prod = pd.DataFrame(index=data2.columns)
    df_prod['Current'] = data2.iloc[-1]
    df_prod['LastWeek'] = data2.iloc[-1] - data2.iloc[-2]
    df_prod['LastYear'] = data2.iloc[-1] - data2.iloc[-52:].mean()
    df_prod['Last5Year'] = data2.iloc[-1] - data2.iloc[-52 * 5:].mean()

    data3 = pd.read_excel('http://ir.eia.gov/wpsr/psw02.xls', 'Data 1', header=1, skiprows=range(2, 3))
    data3.rename(columns={'Sourcekey': 'Date'}, inplace=True)
    data3.set_index('Date', inplace=True)
    data3 = data3[['WCRRIUS2', 'WPULEUS3', 'W_EPM0F_YPR_NUS_MBBLD', 'WDIRPUS2']]
    data3.columns = ['Crude Input to Refineries', 'Refinery Capacity Utilization', 'Gasoline Production', 'Distillate Production']
    data3['Crude Input to Refineries'] = data3['Crude Input to Refineries'] / 1000.0
    data3['Gasoline Production'] = data3['Gasoline Production'] / 1000.0
    data3['Distillate Production'] = data3['Distillate Production'] / 1000.0
    df_production = pd.DataFrame(index=data3.columns)
    df_production['Stock'] = data3.iloc[-1]
    df_production['LastWeek'] = data3.iloc[-1] - data3.iloc[-2]
    df_production['LastYear'] = data3.iloc[-1] - data3.iloc[-52:].mean()
    df_production['Last5Year'] = data3.iloc[-1] - data3.iloc[-52 * 5:].mean()

    title = '<h3>Weekly Crude Oil</h3>'
    summary = f'<h5>Crude stock {data.iloc[-1]["Crude"]:.3f}, change {data.iloc[-1]["Crude"]-data.iloc[-2]["Crude"]: .3f}, or {(data.iloc[-1]["Crude"]-data.iloc[-2]["Crude"])/data.iloc[-2]["Crude"]*100.0:.2f}%</h5>'
    body1 = df_stock.to_html(border=None)#.replace('border="1"','')
    body2 = df_prod.to_html(border=None)
    body3 = df_production.to_html(border=None)

    # --------------------------------------- Create and Send out HTML ------------------------------------------------- #

    html_string = f'''
    <html>
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <!--<style>body{{ margin:0 100; background:whitesmoke; }}</style>-->
            <style>body{{ margin:0 100;}}</style>
            <style>
                table {{
                  border-collapse: collapse;
                  border-spacing: 0;
                  width: 50%;
                  border: 1px solid #ddd;
                }}
                
                th, td {{
                  text-align: left;
                  padding: 16px;
                }}
                
                tr:nth-child(even) {{
                  background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <div>{title}</div>
            <div>{summary}</div>
            <div>{body1}</div>
            <div>{body2}</div>
            <div><h5>Prod+NX-Stock+Adj=Refinery; stock*7=table1</h5></div>
            <div>{body3}</div>
            <div><h5>XB Production+NX-stock==>consumption</div>
        </body>
    </html>'''

    return html_string