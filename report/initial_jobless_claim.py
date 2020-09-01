import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --------------------------------------------- Create contents ----------------------------------------------------- #
# 8:30 a.m. (Eastern Time) on Thursday
def generate_html(today):
    url = 'https://tradingeconomics.com/united-states/jobless-claims'
    page = requests.get(url)
    content = page.content.decode('utf-8')
    soup = BeautifulSoup(page.content, 'html.parser')

    cols = ['Calendar', 'GMT', 'Reference', 'Actual', 'Previous', 'Consensus', 'TEForecast']
    df = pd.DataFrame(columns=cols)
    row = {}
    i = 0
    for td in soup.find_all("td"):
        txt = td.get_text().strip()
        try:
            dt = datetime.strptime(txt, '%Y-%m-%d')
            if len(row) > 0:
                df_row = pd.DataFrame(row, index=[row['Calendar']])
                df = pd.concat([df, df_row], axis=0)
            row['Calendar'] = dt
            i = 1
        except:  # dt is not date
            if 'initial jobless claims' in txt.lower():
                continue
            if i < 1:
                continue
            try:
                row[cols[i]] = txt
                i = i + 1
            except:
                break
    df.set_index('Calendar', inplace=True)

    release_date = df[df.Actual != ''].index[-1]

    if release_date.date() != today.date():
        return None

    title = '<h3>Weekly Initial Jobless Claim</h3>'
    summary = f'<h5>Actual {df[df.Actual != ""].Actual.iloc[-1]}, forecast {df[df.Actual != ""].Consensus.iloc[-1]}, previous {df[df.Actual != ""].Previous.iloc[-1]}</h5>'
    body = df.to_html(border=None)#.replace('border="1"','')

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
            <div>{body}</div>
        </body>
    </html>'''

    return html_string