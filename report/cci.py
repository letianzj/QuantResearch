import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# --------------------------------------------- Create contents ----------------------------------------------------- #
# The Consumer Board's index is released at 10:00 a.m. on the last Tuesday of the month.
def generate_html(today):
    url = 'https://www.ceicdata.com/en/united-states/consumer-confidence-index/consumer-confidence-index'
    page = requests.get(url)
    content = page.content.decode('utf-8')
    soup = BeautifulSoup(page.content, 'html.parser')

    table = soup.find_all('table')[0]
    df = pd.read_html(str(table))[0]

    release_month = df.iloc[0, 1].split(' ')[-2]

    if release_month != today.strftime("%B"):      # monthly update
        return None

    title = '<h3>Monthly Consumer Confidence</h3>'
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