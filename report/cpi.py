import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# --------------------------------------------- Create contents ----------------------------------------------------- #
# From BLS, monthly, begining of next month. one-month seasonal-adjustment cpi
def generate_html(today):
    url = 'https://www.bls.gov/feed/bls_latest.rss'
    page = requests.get(url)
    content = page.content.decode('utf-8')
    soup = BeautifulSoup(page.content, 'lxml')

    ss = soup.find_all('span')[0].text
    ss_month = ss.split(' ')[-2]

    first = today.replace(day=1)
    lastMonth = first - timedelta(days=1)
    lastMonth = lastMonth.strftime("%B")

    if ss_month != lastMonth:
        return None

    title = '<h3>Consumer Price Index, seasonally adjusted</h3>'
    body = ss

    # url = 'https://rigcount.bakerhughes.com/na-rig-count'
    # page = requests.get(url)
    # content = page.content.decode('utf-8')
    # soup = BeautifulSoup(page.content, 'html.parser')
    # table = soup.find_all('table')[0]
    # # df = pd.read_html(str(table))
    # link = table.findAll('a')[0]
    # link = link.get('href')
    # print(link)
    # data = pd.read_excel(link, sheet_name='Current Weekly Summary')       # not working, it's a redirect

    # --------------------------------------- Create and Send out HTML ------------------------------------------------- #

    html_string = f'''
    <html>
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <!--<style>body{{ margin:0 100; background:whitesmoke; }}</style>-->
            <style>body{{ margin:0 100;}}</style>
            <style>

                table {{numpy.ndarray.putmask
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
            <div>{body}</div>
            <div><a href="https://fred.stlouisfed.org/series/CPIAUCSL">Historical from Fred</a> </div>
        </body>
    </html>'''

    return html_string