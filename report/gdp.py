import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --------------------------------------------- Create contents ----------------------------------------------------- #
# GDP quarterly, 8:30am, First reading released first month following the target quarter (e.g. October for the 3rd quarter)
# Revised each of the next two months (Nov Second reading and Dec the final reading)
def generate_html(today):
    url = 'https://tradingeconomics.com/united-states/gdp-growth'
    page = requests.get(url)
    content = page.content.decode('utf-8')
    soup = BeautifulSoup(page.content, 'html.parser')

    cols = ['Calendar', 'GMT', 'Reference', 'Quarter', 'Actual', 'Previous', 'Consensus', 'TEForecast']
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
            if i < 1:
                continue
            try:
                row[cols[i]] = txt
                i = i + 1
            except:
                break
    df.set_index('Calendar', inplace=True)

    release_date = df[df.Actual != ''].index[-1]

    if release_date.month < today.month:      # monthly update
        return None

    title = '<h3>Quarterly GDP</h3>'
    body = df.to_html(border=None)  # .replace('border="1"','')

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
            <div>{body}</div>
        </body>
    </html>'''

    return html_string