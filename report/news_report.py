import os
import argparse
import numpy as np
import pandas as pd
import time
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import plotly
import plotly.graph_objs as go

import eia_crude
import eia_ng
import initial_jobless_claim
import cot_report
import rig_count
import gdp
import retail_sales
import nonfarm_payroll
import cpi
# import cci

list_schedule = [
    ('EIA Weekly Crude', 'eia_crude', 'W', 2, '10:30:00', '10:45:00', True),           # Wed 10:30 except https://www.eia.gov/petroleum/supply/weekly/schedule.php
    ('EIA Weekly NG', 'eia_ng', 'W', 3, '10:30:00', '10:45:00', True),                 # Thu 10:30 except http://ir.eia.gov/ngs/schedule.html
    ('Weekly Initial Jobless Claim', 'initial_jobless_claim', 'W', 3, '08:30:00', '08:45:00', True),    # Thu 8:30 https://tradingeconomics.com/united-states/jobless-claims
    ('COT Report', 'cot_report', 'W', 0, '15:30:00', '15:45:00', True),                # Fri 15:30 https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm
    ('Baker Hughes Rig Count', 'rig_count', 'W', 4, '13:00:00', '13:15:00', True),     # 1pm last day of work week
    ('GDP Quarterly', 'gdp', 'M', 30, '08:30:00', '08:45:00', True),                     # eom, https://www.bea.gov/news/schedule
    ('Retail Sales', 'retail_sales', 'M', 16, '08:30:00', '08:45:00', True),             # around the 13th of every month, https://tradingeconomics.com/united-states/retail-sales
    ('Non-Farm Payroll', 'nonfarm_payroll', 'M', 2, '08:30:00', '08:45:00', True),      # 8:30 a.m. Eastern Time on the first Friday of every month; https://tradingeconomics.com/united-states/non-farm-payrolls
    ('Consumer Price Index', 'cpi', 'M', 14, '08:30:00', '08:45:00', True),              # 8:30 a.m. Eastern Time between 10~15 next month, https://www.bls.gov/schedule/news_release/cpi.htm
    # ('Consumer Confidence', 'cci', 'M', 26, '10:00:00', '10:15:00', True),             # 10:00 a.m. on the last Tuesday of the month
]
labels = ['name', 'module', 'frequency', 'dayofweek', 'starttime', 'endtime', 'available']    #  Mon 0, Tue 1, Wed 2 Thu 3 Fri 4 Sat 5 Sun 6
df_schedule = pd.DataFrame.from_records(list_schedule, columns=labels)

def send_email(email_subject, email_body):
    try:
        #The mail addresses and password
        sender_address = 'contact1@gmail.com'
        sender_pass = 'contact1'
        receiver_address = 'contact1@gmail.com, contact2@gmail.com'
        #Setup the MIME
        message = MIMEMultipart('alternative')
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Subject'] = email_subject
        message.attach(MIMEText(email_body, 'html'))
        #Create SMTP session for sending the mail
        server = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
        server.ehlo()
        server.starttls() #enable security
        server.login(sender_address, sender_pass) #login with mail_id and password
        server.sendmail(sender_address, receiver_address.split(","), message.as_string())
        server.close()
        print('Mail Sent')
    except Exception as e:
        print(f'Gmail error: {str(e)}')


def reset_new_day():
    global df_schedule
    df_schedule['available'] = True


def run(args):
    global  df_schedule
    today = datetime.today()

    # email_body = cpi.generate_html(today)
    # send_email('cpi', email_body)

    while True:
        now_time = today.time().strftime("%H:%M:%S")
        for idx, row in df_schedule.iterrows():
            now = datetime.now()
            now_time = now.time().strftime("%H:%M:%S")

            if today.date() != now.date():   # next day
                reset_new_day()
                today = now

            if now_time >= row.starttime and now_time <= row.endtime and row.available \
                    and (((now.weekday() == row.dayofweek) & (row.frequency == 'W')) or ((now.day == row.dayofweek) & (row.frequency == 'M')) ):
                try:
                    email_body = eval(f'{row.module}.generate_html(now)')
                except Exception as e:
                    print(str(e))
                    email_body = None

                if email_body is not None:
                    df_schedule.loc[idx, 'available'] = False
                    print(f'{row["name"]} received; set to done')
                    send_email(row['name'], email_body)
                else:
                    print(f'{row["name"]} not yet released')
            else:
                print(f'{row["name"]} skipped')

        print(f'sleep 60s {now_time}')
        time.sleep(60.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='news report')
    parser.add_argument('--eiacl', action='store_true')
    parser.add_argument('--sendemail', action='store_true')    # default False
    args = parser.parse_args()

    run(args)