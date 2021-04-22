#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import scipy, scipy.stats
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn import linear_model
import h5py

import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table as dt
import dash_table.FormatTemplate as FormatTemplate
import ta

import data_loader
from futures_tools import get_futures_chain, get_seasonal_contracts
from time_tools import convert_date_input

from app import app
app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally = True

# -------------------------------------------------------- data preparation ---------------------------------------- $
futures_meta_df, futures_contracts_meta_df, inter_comdty_spread_meta_df, inter_comdty_spread_contracts_meta_df = data_loader.load_futures_meta_data()
futures_hist_prices_dict, _ = data_loader.load_futures_hist_prices()
generic_futures_hist_prices_dict = data_loader.load_comdty_generic_hist_prices()
inter_comdty_spread_hist_data_dict = data_loader.load_inter_comdty_spread_hist_prices()
generic_inter_comdty_hist_prices_dict = data_loader.load_inter_comdty_generic_hist_prices()
spread_scores_dict = data_loader.load_spread_score()
fly_scores_dict = data_loader.load_fly_score()

cols = ['Name', 'Contract', 'Price', 'Chg', '1MChg', 'High', 'Low', 'Avg', 'SD', 'EWMA', 'PCT', 'z-score', 'RSI', 'MACD', 'MACDSignal', 'MACDHist', 'MACDHistMin', 'MACDHistMax', 'BBLower', 'BBMid', 'BBUpper']
cols_spread = ['Name', 'Leg1', 'Leg2', 'Leg1 Actual', 'Leg2 Actual', 'Spread', 'Spread Prcnt', 'Spread Z-Score', 'RD Prcnt', 'RD Z-Score']
cols_fly = ['Name', 'Leg1', 'Leg2', 'Leg3', 'Leg1 Actual', 'Leg2 Actual', 'Leg3 Actual', 'Fly', 'Fly Prcnt', 'Fly Z-Score', 'RD Prcnt','RD Z-Score']

df_single = pd.DataFrame(columns=cols)
for root_sym in generic_futures_hist_prices_dict.keys():
    try:
        hist_series = generic_futures_hist_prices_dict[root_sym][root_sym + '1'].copy()
        hist_series.dropna(inplace=True)
        hist_series = hist_series[(hist_series.index[-1]+timedelta(days=-365*5)):]      # last 5 years

        meta_data = futures_contracts_meta_df[futures_contracts_meta_df['Root']==root_sym]
        meta_data.sort_values('Last_Trade_Date', inplace=True)

        row_dict = {}
        row_dict['Name'] = futures_meta_df.loc[root_sym, "NAME"][:-6]
        row_dict['Contract'] = get_futures_chain(meta_data, hist_series.index[-1]).index[0]
        row_dict['Price'] = round(hist_series.iloc[-1], 4)
        row_dict['Chg'] = round((hist_series.iloc[-1] / hist_series.iloc[-2] - 1.0) * 100.0, 4)
        try:
            row_dict['1MChg'] = round((hist_series.iloc[-1] / hist_series.iloc[-22] - 1.0) * 100.0, 4)
        except:
            row_dict['1MChg'] = None
        row_dict['High'] = round(np.max(hist_series.dropna().to_numpy()), 4)
        row_dict['Low'] = round(np.min(hist_series.dropna().to_numpy()), 4)
        row_dict['Avg'] = round(np.average(hist_series.dropna().to_numpy()), 4)
        row_dict['SD'] = round(np.std(hist_series.dropna().to_numpy()), 4)
        hist_return = hist_series / hist_series.shift(1) - 1
        row_dict['EWMA'] = round(np.sqrt((hist_return.sort_index(ascending=False)**2).ewm(alpha=0.06).mean()[0] * 252), 4)
        row_dict['PCT'] = round(scipy.stats.percentileofscore(hist_series.dropna().to_numpy(), hist_series.iloc[-1]), 4)
        row_dict['z-score'] = round((row_dict['Price'] - row_dict['Avg']) / row_dict['SD'], 4)
        row_dict['RSI'] = round(ta.momentum.rsi(hist_series, window=14*2-1)[-1], 4)
        row_dict['MACD'] = round(ta.trend.macd(hist_series)[-1], 4)
        row_dict['MACDSignal'] = round(ta.trend.macd_signal(hist_series)[-1], 4)
        row_dict['MACDHist'] = round(ta.trend.macd_diff(hist_series)[-1], 4)
        row_dict['MACDHistMin'] = round(np.nanmin(row_dict['MACDHist']), 4)
        row_dict['MACDHistMax'] = round(np.nanmax(row_dict['MACDHist']), 4)
        row_dict['BBLower'] = round(ta.volatility.bollinger_lband(hist_series)[-1], 4)
        row_dict['BBMid'] = round(ta.volatility.bollinger_mavg(hist_series)[-1], 4)
        row_dict['BBUpper'] = round(ta.volatility.bollinger_hband(hist_series)[-1], 4)
        df_temp = pd.DataFrame(row_dict, index=[root_sym])
        df_single = df_single.append(df_temp)
    except:
        pass

df_single = df_single[cols]
#df = df.sort_values(by=['z-score'])

df_inter = pd.DataFrame(columns=cols)
for root_sym in generic_inter_comdty_hist_prices_dict.keys():
    try:
        hist_series = generic_inter_comdty_hist_prices_dict[root_sym][root_sym + '1'].copy()
        hist_series.dropna(inplace=True)
        hist_series = hist_series[(hist_series.index[-1]+timedelta(days=-365*5)):]      # last 5 years

        meta_data = inter_comdty_spread_contracts_meta_df[inter_comdty_spread_contracts_meta_df['Root']==root_sym]
        meta_data.sort_values('Last_Trade_Date', inplace=True)

        row_dict = {}
        row_dict['Name'] = root_sym
        row_dict['Contract'] = get_futures_chain(meta_data, hist_series.index[-1]).index[0]
        row_dict['Price'] = round(hist_series.iloc[-1], 4)
        row_dict['Chg'] = round((hist_series.iloc[-1] / hist_series.iloc[-2] - 1.0) * 100.0, 4)
        try:
            row_dict['1MChg'] = round((hist_series.iloc[-1] / hist_series.iloc[-22] - 1.0) * 100.0, 4)
        except:
            row_dict['1MChg'] = None
        row_dict['High'] = round(np.max(hist_series.dropna().to_numpy()), 4)
        row_dict['Low'] = round(np.min(hist_series.dropna().to_numpy()), 4)
        row_dict['Avg'] = round(np.average(hist_series.dropna().to_numpy()), 4)
        row_dict['SD'] = round(np.std(hist_series.dropna().to_numpy()), 4)
        hist_return = hist_series / hist_series.shift(1) - 1
        row_dict['EWMA'] = round(
            np.sqrt((hist_return.sort_index(ascending=False) ** 2).ewm(alpha=0.06).mean()[0] * 252), 4)
        row_dict['PCT'] = round(scipy.stats.percentileofscore(hist_series.dropna().to_numpy(), hist_series.iloc[-1]),
                                4)
        row_dict['z-score'] = round((row_dict['Price'] - row_dict['Avg']) / row_dict['SD'], 4)
        row_dict['RSI'] = round(ta.momentum.rsi(hist_series, window=14*2-1)[-1], 4)
        row_dict['MACD'] = round(ta.trend.macd(hist_series)[-1], 4)
        row_dict['MACDSignal'] = round(ta.trend.macd_signal(hist_series)[-1], 4)
        row_dict['MACDHist'] = round(ta.trend.macd_diff(hist_series)[-1], 4)
        row_dict['MACDHistMin'] = round(np.nanmin(row_dict['MACDHist']), 4)
        row_dict['MACDHistMax'] = round(np.nanmax(row_dict['MACDHist']), 4)
        row_dict['BBLower'] = round(ta.volatility.bollinger_lband(hist_series)[-1], 4)
        row_dict['BBMid'] = round(ta.volatility.bollinger_mavg(hist_series)[-1], 4)
        row_dict['BBUpper'] = round(ta.volatility.bollinger_hband(hist_series)[-1], 4)
        df_temp = pd.DataFrame(row_dict, index=[root_sym])
        df_inter = df_inter.append(df_temp)
    except:
        pass

df_inter = df_inter[cols]
#df = df.sort_values(by=['z-score'])

# -------------------------------------------------------- help functions ---------------------------------------- $

# -------------------------------------------------------- define layout ---------------------------------------- $
# app.css.append_css({"external_url": 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

layout = html.Div([
    html.Div([
        html.H2("Commodity Futures")
    ], className='banner'),

    html.Div(id='cached-data-market-commodity-futures-tab1', style={'display': 'none'}),
    html.Div(id='cached-data-market-commodity-futures-tab2', style={'display': 'none'}),
    html.Div(id='cached-data-market-commodity-futures-tab3', style={'display': 'none'}),
    html.Div(id='cached-data-market-commodity-futures-tab4', style={'display': 'none'}),

    html.Div([
        dcc.Tabs(
            children=[
                dcc.Tab(label='Single Futures', value=1),
                dcc.Tab(label='Curve Spread', value=2),
                dcc.Tab(label='Curve Fly', value=3),
                dcc.Tab(label='Seasonality', value=4),
            ],
            value=1,
            id='market-commodity-futures-tabs',
        ),
        html.Div(id='market-commodity-futures-tab-output')
    ]),

    html.Div(id='hidden-div-market-commodity-futures', style={'display': 'none'})

    ], style={
    'width': '90%',
    'fontFamily': 'Sans-Serif',
    'margin-left': 'auto',
    'margin-right': 'auto'}
)

# -------------------------------------------------------- Tab Layout -------------------------- #
@app.callback(
    Output('market-commodity-futures-tab-output', 'children'),
    [Input('market-commodity-futures-tabs', 'value')])
def update_tabs_market_commodity_futures(tab_choice):
    if tab_choice == 1:
        return \
html.Div([
    html.Div([], className='twelve columns wind-polar'),  # seems that this is needed for alignment)

    html.Div([
        dcc.RadioItems(
            id='outright-spread-market-commodity-futures-tab1',
            options=[
                {'label': 'Outright', 'value': 'Outright'},
                {'label': 'InterComdty', 'value': 'InterComdty'}
            ],
            value='Outright'
        )
    ], className='four columns wind-polar'),

    html.Div([
        dt.DataTable(
            style_table={'overflowX': 'scroll'},
            columns=[{"name": i, "id": i, "deletable": False} for i in cols],
            editable=False,
            row_deletable=False,
            filter_action="native",
            sort_action="native",
            sort_mode='multi',
            row_selectable='single',  # multi
            selected_rows=[0],
            page_action='native',
            page_current=0,
            page_size=15,
            id='overview-table-market-commodity-futures-tab1'
        )
    ], className='twelve columns row wind-speed-row'),

    # html.Div([
    #     html.Div([
    #         html.Div([html.H3('Historical Time Series')], className='Title'),
    #         html.Div([dcc.Graph(id='macro-data-explorer-historical-time-series')]),
    #     ], className='six columns wind-polar'),
    #     html.Div([
    #         html.Div([html.H3('Term Structure Curve')], className='Title'),
    #         html.Div([dcc.Graph(id='macro-data-explorer-term-structure')]),
    #     ], className='six columns wind-polar')
    # ], className='row wind-speed-row')

    html.Div([html.H3('Historical Generic Prices')], className='Title twelve columns wind-polar'),

    html.Div([
        html.Div([
            dcc.Input(
                id='generic-series-start-number-market-commodity-futures-tab1',
                placeholder='Enter number of series',
                type='text',
                value=''
            )
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(
                id='generic-series-end-number-market-commodity-futures-tab1',
                placeholder='Enter number of series',
                type='text',
                value=''
            )
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(id='lookback-selection-market-commodity-futures-tab1', placeholder='Lookback (yyyy-mm-dd) or 5Y',
                      type='text', value='')
        ], className='two columns wind-polar'),

        html.Div([
            html.Button('Go', id='historical-generic-series-button-market-commodity-futures-tab1')
        ], className='two columns wind-polar'),
    ], className='twelve columns wind-polar'),

    html.Div([
        dcc.Graph(id='historical-time-series-market-commodity-futures-tab1')
    ], className='twelve columns row wind-speed-row'),

    html.Div([html.H3('Term Structures')], className='Title twelve columns wind-polar'),

    html.Div([
        html.Div([
            dcc.Input(
                id='term-structure-date-one-market-commodity-futures-tab1',
                placeholder='Enter date (-5y, yyyy-mm-dd)',
                type='text',
                value=''
            )
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(
                id='term-structure-date-two-market-commodity-futures-tab1',
                placeholder='Enter date (-5y, yyyy-mm-dd)',
                type='text',
                value=''
            )
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(
                id='term-structure-date-three-market-commodity-futures-tab1',
                placeholder='Enter date (-5y, yyyy-mm-dd)',
                type='text',
                value=''
            )
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(
                id='term-structure-date-four-market-commodity-futures-tab1',
                placeholder='Enter date (-5y, yyyy-mm-dd)',
                type='text',
                value=''
            )
        ], className='two columns wind-polar'),

        html.Div([
            dcc.Input(
                id='term-structure-date-five-market-commodity-futures-tab1',
                placeholder='Enter date (-5y, yyyy-mm-dd)',
                type='text',
                value=''
            )
        ], className='two columns wind-polar'),

        html.Div([
            html.Button('Go', id='term-structure-button-market-commodity-futures-tab1')
        ], className='two columns wind-polar'),
    ], className='twelve columns wind-polar'),

    html.Div([
        dcc.Graph(style={'height': '550px'}, id='historical-term-structures-market-commodity-futures-tab1')
    ], className='twelve columns row wind-speed-row'),
])

    elif tab_choice == 2:
        return html.Div([
            html.Div([
                dcc.Dropdown(
                    id='product-dropdown-curve-spread-market-commodity-futures-tab2',
                    options=[
                        {'label': sym_root, 'value': sym_root} for sym_root in spread_scores_dict.keys()
                    ],
                    value='CL'
                )
            ], className='four columns wind-polar',
                style={'width': '10%', 'display': 'inline-block', 'padding-bottom': '1%', 'horizontal-align': 'top'}),

            html.Div([
                dcc.Input(id='lookback-selection-market-commodity-futures-tab2', placeholder='Lookback (yyyy-mm-dd) or 5Y',
                          type='text', value='')
            ], className='two columns wind-polar'),

            html.Div([
                dt.DataTable(
                    style_table={'overflowX': 'scroll'},
                    columns=[{"name": i, "id": i, "deletable": False} for i in cols_spread],
                    editable=False,
                    row_deletable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode='multi',
                    row_selectable='single',  # multi
                    selected_rows=[0],
                    page_action='native',
                    page_current=0,
                    page_size=15,
                    id='spread-score-table-market-commodity-futures-tab2'
                )
            ], className='twelve columns row wind-speed-row'),

            html.Div([
                html.Div([
                    dcc.Graph(style={'height': '450px'}, id='historical-spread-time-series-market-commodity-futures-tab2')
                ], className='six columns wind-polar'),

                html.Div([
                    dcc.Graph(style={'height': '450px'},  id='historical-spread-scatterplot-market-commodity-futures-tab2')
                ], className='six columns wind-polar'),
            ], 'twelve columns row wind-speed-row')
        ])

    elif tab_choice == 3:
        return html.Div([
            html.Div([
                dcc.Dropdown(
                    id='product-dropdown-curve-fly-market-commodity-futures-tab3',
                    options=[
                        {'label': sym_root, 'value': sym_root} for sym_root in fly_scores_dict.keys()
                    ],
                    value='CL'
                )
            ], className='four columns wind-polar',
                style={'width': '10%', 'display': 'inline-block', 'padding-bottom': '1%', 'horizontal-align': 'top'}),

            html.Div([
                dcc.Input(id='lookback-selection-market-commodity-futures-tab3',
                          placeholder='Lookback (yyyy-mm-dd) or 5Y',
                          type='text', value='')
            ], className='two columns wind-polar'),

            html.Div([
                dt.DataTable(
                    style_table={'overflowX': 'scroll'},
                    columns=[{"name": i, "id": i, "deletable": False} for i in cols_fly],
                    editable=False,
                    row_deletable=False,
                    filter_action="native",
                    sort_action="native",
                    sort_mode='multi',
                    row_selectable='single',  # multi
                    selected_rows=[0],
                    page_action='native',
                    page_current=0,
                    page_size=15,
                    id='fly-score-table-market-commodity-futures-tab3'
                )
            ], className='twelve columns row wind-speed-row'),

            html.Div([
                html.Div([
                    dcc.Graph(style={'height': '450px'}, id='historical-fly-time-series-market-commodity-futures-tab3')
                ], className='six columns wind-polar'),

                html.Div([
                    dcc.Graph(style={'height': '450px'}, id='historical-fly-scatterplot-market-commodity-futures-tab3')
                ], className='six columns wind-polar'),
            ], className='twelve columns row wind-speed-row')
        ])
    elif tab_choice == 4:
        return html.Div([
            html.Div([html.H3('Seasonality')], className='Title twelve columns wind-polar'),

            html.Div([
                html.Div([
                    dcc.Input(
                        id='seasonality-contract-one-market-commodity-futures-tab4',
                        placeholder='Enter contract (e.g. NGZ2020))',
                        type='text',
                        value=''
                    )
                ], className='two columns wind-polar'),

                html.Div([
                    dcc.Input(
                        id='seasonality-contract-two-market-commodity-futures-tab4',
                        placeholder='Enter contract (e.g. NGZ2020))',
                        type='text',
                        value=''
                    )
                ], className='two columns wind-polar'),

                html.Div([
                    dcc.Input(
                        id='seasonality-contract-three-market-commodity-futures-tab4',
                        placeholder='Enter contract (e.g. NGZ2020))',
                        type='text',
                        value=''
                    )
                ], className='two columns wind-polar'),
            ], className='twelve columns wind-polar'),

            html.Div([
                html.Div([
                    dcc.Input(
                        id='seasonality-weight-one-market-commodity-futures-tab4',
                        placeholder='Enter weight (e.g. 1)',
                        type='text',
                        value=''
                    )
                ], className='two columns wind-polar'),

                html.Div([
                    dcc.Input(
                        id='seasonality-weight-two-market-commodity-futures-tab4',
                        placeholder='Enter weight (e.g. -2)',
                        type='text',
                        value=''
                    )
                ], className='two columns wind-polar'),

                html.Div([
                    dcc.Input(
                        id='seasonality-weight-three-market-commodity-futures-tab4',
                        placeholder='Enter weight (e.g. 1)',
                        type='text',
                        value=''
                    )
                ], className='two columns wind-polar'),
            ], className='twelve columns wind-polar'),

            html.Div([
                html.Div([
                    dcc.Input(
                        id='seasonality-lookback-window-market-commodity-futures-tab4',
                        placeholder='Enter lookback days (e.g. 250)',
                        type='text',
                        value=''
                    )
                ], className='two columns wind-polar'),

                html.Div([
                    html.Button('Go', id='seasonality-button-market-commodity-futures-tab4')
                ], className='two columns wind-polar'),
            ], className='twelve columns wind-polar'),

            html.Div([
                dcc.Graph(style={'height': '700px'}, id='seasonal-term-structures-market-commodity-futures-tab4')
            ], className='twelve columns row wind-speed-row'),
        ])


# -------------------------------------------------------- define event handler -------------------------------------- $
@app.callback(
    Output('overview-table-market-commodity-futures-tab1', 'data'),
    [Input('outright-spread-market-commodity-futures-tab1', 'value')]
)
def update_overview_table_market_commodity_futures_tab1(inter_comdty):
    if inter_comdty == 'Outright':
        return df_single.to_dict('records')
    else:
        return df_inter.to_dict('records')


@app.callback(
    Output('historical-time-series-market-commodity-futures-tab1', 'figure'),
    [Input('overview-table-market-commodity-futures-tab1', 'data'),
     Input('overview-table-market-commodity-futures-tab1', 'selected_rows'),
     Input('historical-generic-series-button-market-commodity-futures-tab1', 'n_clicks')],
    [State('generic-series-start-number-market-commodity-futures-tab1', 'value'),
     State('generic-series-end-number-market-commodity-futures-tab1', 'value'),
     State('lookback-selection-market-commodity-futures-tab1', 'value')]
)
def update_historical_time_series_market_commodity_futures_tab1(rows_data, rows_selected, n_clicks, generic_start_str, generic_end_str, lookback_window):
    # print('historical series called')
    sym_root = rows_data[rows_selected[0]]['Contract'][:-5]        # remove e.g. Z2018
    # sym = df_config.loc[df_config['Name'] == name]['Quandl Code'].values[0]
    lookback_date = convert_date_input(lookback_window, datetime(2008, 1, 1))

    if ':' in sym_root:
        df = generic_inter_comdty_hist_prices_dict[sym_root]
    else:
        df = generic_futures_hist_prices_dict[sym_root]

    df = df[lookback_date.date():]

    generic_start = 1
    if (generic_start_str is not None) and (not not generic_start_str):
        generic_start = int(generic_start_str)
    generic_end = df.shape[1]
    if (generic_end_str is not None) and (not not generic_end_str):
        generic_end = int(generic_end_str)

    traces = [go.Scatter(x=df[col].index,
                         y=df[col],
                         mode='lines',
                         name=col)
              for col in df.columns[generic_start-1:generic_end]]

    layout_fig = go.Layout(
        xaxis=dict(title='Date',
                   rangeslider=dict(
                       visible=False
                   ),
                   type='date'),
        yaxis=dict(title='Price'),
        legend=dict(orientation="h"),
        height=800, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return go.Figure(data=traces, layout=layout_fig)


@app.callback(
    Output('historical-term-structures-market-commodity-futures-tab1', 'figure'),
    [Input('overview-table-market-commodity-futures-tab1', 'data'),
     Input('overview-table-market-commodity-futures-tab1', 'selected_rows'),
     Input('term-structure-button-market-commodity-futures-tab1', 'n_clicks')],
    [State('term-structure-date-one-market-commodity-futures-tab1', 'value'),
     State('term-structure-date-two-market-commodity-futures-tab1', 'value'),
     State('term-structure-date-three-market-commodity-futures-tab1', 'value'),
     State('term-structure-date-four-market-commodity-futures-tab1', 'value'),
     State('term-structure-date-five-market-commodity-futures-tab1', 'value')]
)
def update_historical_term_structures_market_commodity_futures_tab1(rows_data, rows_selected, n_clicks, ione, itwo, ithree, ifour, ifive):
    # print('historical term structure called')
    sym_root = rows_data[rows_selected[0]]['Contract'][:-5]

    if ':' in sym_root:
        hist_data = inter_comdty_spread_hist_data_dict[sym_root]
        meta_data = inter_comdty_spread_contracts_meta_df[inter_comdty_spread_contracts_meta_df['Root'] == sym_root]
        meta_data.sort_values('Last_Trade_Date', inplace=True)
    else:
        hist_data = futures_hist_prices_dict[sym_root]
        meta_data = futures_contracts_meta_df[futures_contracts_meta_df['Root'] == sym_root]
        meta_data.sort_values('Last_Trade_Date', inplace=True)

    asofdate = hist_data.index[-1]
    s0 = hist_data.loc[asofdate]
    s = s0.to_frame()

    start_idx = hist_data.shape[0] - 1
    if (ione is not None) and (not not ione):
        t1 = convert_date_input(ione, datetime.today())
        t1 = t1.date()
        dateidx1 = hist_data.index.searchsorted(t1)  # first one greater than or equal to
        s1 = hist_data.iloc[dateidx1]
        s = pd.concat([s, s1], axis=1)
        start_idx = min(dateidx1, start_idx)

    if (itwo is not None) and (not not itwo):
        t2 = convert_date_input(itwo, datetime.today())
        t2 = t2.date()
        dateidx2 = hist_data.index.searchsorted(t2)  # first one greater than or equal to
        s2 = hist_data.iloc[dateidx2]
        s = pd.concat([s, s2], axis=1)
        start_idx = min(dateidx2, start_idx)

    if (ithree is not None) and (not not ithree):
        t3 = convert_date_input(ithree, datetime.today())
        t3 = t3.date()
        dateidx3 = hist_data.index.searchsorted(t3)  # first one greater than or equal to
        s3 = hist_data.iloc[dateidx3]
        s = pd.concat([s, s3], axis=1)
        start_idx = min(dateidx3, start_idx)

    if (ifour is not None) and (not not ifour):
        t4 = convert_date_input(ifour, datetime.today())
        t4 = t4.date()
        dateidx4 = hist_data.index.searchsorted(t4)  # first one greater than or equal to
        s4 = hist_data.iloc[dateidx4]
        s = pd.concat([s, s4], axis=1)
        start_idx = min(dateidx4, start_idx)

    if (ifive is not None) and (not not ifive):
        t5 = convert_date_input(ifive, datetime.today())
        t5 = t5.date()
        dateidx5 = hist_data.index.searchsorted(t5)  # first one greater than or equal to
        s5 = hist_data.iloc[dateidx5]
        s = pd.concat([s, s5], axis=1)
        start_idx = min(dateidx5, start_idx)

    st = s.join(meta_data['Last_Trade_Date'], how='left')
    st = st.sort_values('Last_Trade_Date')

    # find the first common date
    # dateidx_st = st['Last_Trade_Date'].searchsorted(hist_data.index[start_idx])[0]
    dateidx_st = st['Last_Trade_Date'].searchsorted(hist_data.index[start_idx])
    st = st.iloc[dateidx_st:]
    # st.fillna(0.0, inplace=True)
    traces = [go.Scatter(x=st['Last_Trade_Date'], y=st[c], name=c.strftime('%Y-%m-%d'), mode='lines+markers', hovertext=st.index) for c in st.columns[:-1]]
    layout_fig = go.Layout(title=sym_root, xaxis={'title': sym_root}, yaxis={'title': 'Price'},
                           legend=dict(orientation="h"),
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)'
                           )

    #plotly.offline.plot({'data': traces, 'layout': layout})
    return go.Figure(data=traces, layout=layout_fig)


@app.callback(
    Output('spread-score-table-market-commodity-futures-tab2', 'data'),
    [Input('product-dropdown-curve-spread-market-commodity-futures-tab2', 'value')]
)
def update_spread_score_table_market_commodity_futures_tab2(sym_root):
    df = spread_scores_dict[sym_root]
    df = df[cols_spread]
    return df.to_dict('records')


@app.callback(
    Output('historical-spread-time-series-market-commodity-futures-tab2', 'figure'),
    [Input('spread-score-table-market-commodity-futures-tab2', 'data'),
     Input('spread-score-table-market-commodity-futures-tab2', 'selected_rows'),
     Input('lookback-selection-market-commodity-futures-tab2', 'value')]
)
def update_historical_spread_time_series_market_commodity_futures_tab2(rows_data, rows_selected, lookback_window):
    sym_root = rows_data[rows_selected[0]]['Name']
    leg1 = rows_data[rows_selected[0]]['Leg1 Actual']
    leg2 = rows_data[rows_selected[0]]['Leg2 Actual']

    lookback_date = convert_date_input(lookback_window, datetime(2008, 1, 1))

    if ':' in sym_root:
        df1 = inter_comdty_spread_hist_data_dict[sym_root][leg1]
        df2 = inter_comdty_spread_hist_data_dict[sym_root][leg2]
    else:
        df1 = futures_hist_prices_dict[sym_root][leg1]
        df2 = futures_hist_prices_dict[sym_root][leg2]

    df = df1 - df2
    df = df[lookback_date.date():]

    trace = go.Scatter(x=df.index, y=df, name=f'{leg1}-{leg2}', mode='lines')
    layout_fig = go.Layout(title=sym_root, xaxis={'title': sym_root, 'type': 'date', 'tickformat': '%Y-%m-%d'},
                           yaxis={'title': 'Price'}, legend=dict(orientation="h"),
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)'
                           )

    return go.Figure(data=[trace], layout=layout_fig)


@app.callback(
    Output('historical-spread-scatterplot-market-commodity-futures-tab2', 'figure'),
    [Input('spread-score-table-market-commodity-futures-tab2', 'data'),
     Input('spread-score-table-market-commodity-futures-tab2', 'selected_rows'),
     Input('lookback-selection-market-commodity-futures-tab2', 'value')]
)
def update_historical_spread_scatterplot_market_commodity_futures_tab2(rows_data, rows_selected, lookback_window):
    sym_root = rows_data[rows_selected[0]]['Name']
    leg1 = rows_data[rows_selected[0]]['Leg1 Actual']
    leg2 = rows_data[rows_selected[0]]['Leg2 Actual']

    lookback_date = convert_date_input(lookback_window, datetime(2008, 1, 1))

    if ':' in sym_root:
        df1 = inter_comdty_spread_hist_data_dict[sym_root][leg1]
        df2 = inter_comdty_spread_hist_data_dict[sym_root][leg2]
        df0 = generic_inter_comdty_hist_prices_dict[sym_root][sym_root + '1']
    else:
        df1 = futures_hist_prices_dict[sym_root][leg1]
        df2 = futures_hist_prices_dict[sym_root][leg2]
        df0 = generic_futures_hist_prices_dict[sym_root][sym_root + '1']

    df = df1 - df2
    df = pd.concat([df0, df], axis=1)
    df = df[lookback_date.date():]
    df.dropna(inplace=True)

    trace1 = go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], name=f'{leg1}-{leg2}', mode='markers')
    trace2 = go.Scatter(x=[df.iloc[-1, 0]], y=[df.iloc[-1, 1]], name='today', mode='markers',
                        marker=dict(color=['red'], size=[20]))
    layout_fig = go.Layout(xaxis=dict(title='Generic 1st price'),
                           yaxis=dict(title='Spread price'),
                           showlegend=False,
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)'
                           )

    return go.Figure(data=[trace1, trace2], layout=layout_fig)


@app.callback(
    Output('fly-score-table-market-commodity-futures-tab3', 'data'),
    [Input('product-dropdown-curve-fly-market-commodity-futures-tab3', 'value')]
)
def update_fly_score_table_market_commodity_futures_tab3(sym_root):
    df = fly_scores_dict[sym_root]
    df = df[cols_fly]
    return df.to_dict('records')


@app.callback(
    Output('historical-fly-time-series-market-commodity-futures-tab3', 'figure'),
    [Input('fly-score-table-market-commodity-futures-tab3', 'data'),
     Input('fly-score-table-market-commodity-futures-tab3', 'selected_rows'),
     Input('lookback-selection-market-commodity-futures-tab3', 'value')]
)
def update_historical_fly_time_series_market_commodity_futures_tab3(rows_data, rows_selected, lookback_window):
    sym_root = rows_data[rows_selected[0]]['Name']
    leg1 = rows_data[rows_selected[0]]['Leg1 Actual']
    leg2 = rows_data[rows_selected[0]]['Leg2 Actual']
    leg3 = rows_data[rows_selected[0]]['Leg3 Actual']

    lookback_date = convert_date_input(lookback_window, datetime(2008, 1, 1))

    if ':' in sym_root:
        df1 = inter_comdty_spread_hist_data_dict[sym_root][leg1]
        df2 = inter_comdty_spread_hist_data_dict[sym_root][leg2]
        df3 = inter_comdty_spread_hist_data_dict[sym_root][leg3]
    else:
        df1 = futures_hist_prices_dict[sym_root][leg1]
        df2 = futures_hist_prices_dict[sym_root][leg2]
        df3 = futures_hist_prices_dict[sym_root][leg3]

    df = df1 - df2*2 + df3
    df = df[lookback_date.date():]

    trace = go.Scatter(x=df.index, y=df, name=f'{leg1}-{leg2}', mode='lines')
    layout_fig = go.Layout(title=sym_root, xaxis={'title': sym_root, 'type': 'date', 'tickformat': '%Y-%m-%d'},
                           yaxis={'title': 'Price'}, legend=dict(orientation="h"),
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)'
                           )

    return go.Figure(data=[trace], layout=layout_fig)


@app.callback(
    Output('historical-fly-scatterplot-market-commodity-futures-tab3', 'figure'),
    [Input('fly-score-table-market-commodity-futures-tab3', 'data'),
     Input('fly-score-table-market-commodity-futures-tab3', 'selected_rows'),
     Input('lookback-selection-market-commodity-futures-tab3', 'value')]
)
def update_historical_fly_scatterplot_market_commodity_futures_tab3(rows_data, rows_selected, lookback_window):
    sym_root = rows_data[rows_selected[0]]['Name']
    leg1 = rows_data[rows_selected[0]]['Leg1 Actual']
    leg2 = rows_data[rows_selected[0]]['Leg2 Actual']
    leg3 = rows_data[rows_selected[0]]['Leg3 Actual']

    lookback_date = convert_date_input(lookback_window, datetime(2008, 1, 1))

    if ':' in sym_root:
        df1 = inter_comdty_spread_hist_data_dict[sym_root][leg1]
        df2 = inter_comdty_spread_hist_data_dict[sym_root][leg2]
        df3 = inter_comdty_spread_hist_data_dict[sym_root][leg3]
        df0 = generic_inter_comdty_hist_prices_dict[sym_root][sym_root + '1']
    else:
        df1 = futures_hist_prices_dict[sym_root][leg1]
        df2 = futures_hist_prices_dict[sym_root][leg2]
        df3 = futures_hist_prices_dict[sym_root][leg3]
        df0 = generic_futures_hist_prices_dict[sym_root][sym_root + '1']

    df = df1 - df2*2 + df3
    df = pd.concat([df0, df], axis=1)
    df = df[lookback_date.date():]

    trace1 = go.Scatter(x=df.iloc[:, 0], y=df.iloc[:, 1], name=f'{leg1}-{leg2}', mode='markers')
    trace2 = go.Scatter(x=[df.iloc[-1, 0]], y=[df.iloc[-1, 1]], name='today', mode='markers',
                        marker=dict(color=['red'], size=[20]))
    layout_fig = go.Layout(xaxis=dict(title='Generic 1st price'),
                           yaxis=dict(title='Spread price'),
                           showlegend=False,
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)'
                           )

    return go.Figure(data=[trace1, trace2], layout=layout_fig)


@app.callback(
    Output('seasonal-term-structures-market-commodity-futures-tab4', 'figure'),
    [Input('seasonality-button-market-commodity-futures-tab4', 'n_clicks')],
    [State('seasonality-contract-one-market-commodity-futures-tab4', 'value'),
     State('seasonality-contract-two-market-commodity-futures-tab4', 'value'),
     State('seasonality-contract-three-market-commodity-futures-tab4', 'value'),
     State('seasonality-weight-one-market-commodity-futures-tab4', 'value'),
     State('seasonality-weight-two-market-commodity-futures-tab4', 'value'),
     State('seasonality-weight-three-market-commodity-futures-tab4', 'value'),
     State('seasonality-lookback-window-market-commodity-futures-tab4', 'value')]
)
def update_seasonality_curves_market_commodity_futures_tab4(n_clicks, contract1, contract2, contract3, weight1, weight2, weight3, lookback):
    if (contract1 is None) or (not contract1):
        return go.Figure()

    if (contract2 is not None) and (not not contract2):
        if (contract3 is not None) and (not not contract3):
            contracts = [contract1.upper(), contract2.upper(), contract3.upper()]
            weights = [int(weight1), int(weight2), int(weight3)]
        else:
            contracts = [contract1.upper(), contract2.upper()]
            weights = [int(weight1), int(weight2)]
    else:
        contracts = [contract1.upper()]
        weights = [int(weight1)]

    sym_root = contracts[0][:-5]
    if ':' in sym_root:
        hist_data = inter_comdty_spread_hist_data_dict[sym_root]
        meta_data = inter_comdty_spread_contracts_meta_df[inter_comdty_spread_contracts_meta_df['Root'] == sym_root]
        meta_data.sort_values('Last_Trade_Date', inplace=True)
    else:
        hist_data = futures_hist_prices_dict[sym_root]
        meta_data = futures_contracts_meta_df[futures_contracts_meta_df['Root'] == sym_root]
        meta_data.sort_values('Last_Trade_Date', inplace=True)

    nlookback = 5000
    if (lookback is not None) and (not not lookback):
        nlookback = int(lookback)

    asofdate = hist_data.index[-1]
    s = get_seasonal_contracts(asofdate, contracts, weights, hist_data, meta_data)
    s = s.iloc[-nlookback:]
    traces = [go.Scatter(x=s.index, y=s[c], name=c, mode='lines') for c in s.columns]
    layout_fig = go.Layout(title=sym_root, xaxis={'title': sym_root, 'type': 'date', 'tickformat': '%b %d'},
                           yaxis={'title': 'Price'}, legend=dict(orientation="h"),
                           paper_bgcolor='rgba(0,0,0,0)',
                           plot_bgcolor='rgba(0,0,0,0)'
                           )

    return go.Figure(data=traces, layout=layout_fig)