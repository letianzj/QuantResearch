#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table as dt
import logging

from app import app
from futures import commodity_futures_app
from misc import misc_data_app

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(dt.DataTable(data=[{}]), style={'display': 'none'}),
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/market/futures':
        return commodity_futures_app.layout
    elif pathname == '/market/misc':
        return misc_data_app.layout
    else:
        return '404'


if __name__ == '__main__':
    # app.run_server(debug=True)
    app.server.run(host='0.0.0.0',port=5555, debug=False)