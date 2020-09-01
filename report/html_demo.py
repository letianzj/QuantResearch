import os
import pandas as pd
import numpy as np

import plotly
import plotly.graph_objs as go

# --------------------------------------------- Create contents ----------------------------------------------------- #
def generate_html():
    trace1 = go.Scatter(
        x=[1, 2, 3],
        y=[4, 5, 6],
        mode='markers+text',
        text=['Text A', 'Text B', 'Text C'],
        textposition='bottom center'
    )
    trace2 = go.Scatter(
        x=[20, 30, 40],
        y=[50, 60, 70],
        mode='markers+text',
        text=['Text D', 'Text E', 'Text F'],
        textposition='bottom center'
    )

    fig = plotly.subplots.make_subplots(rows=1, cols=2)

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout'].update(height=600, width=800, title='Example Plot')
    div_fig1 = plotly.offline.plot(fig, include_plotlyjs=True, output_type='div')

    data = [['AAPL',10],['GOOGL',12],['AMZN',13]]
    df = pd.DataFrame(data,columns=['Name','RSI'])
    div_table1 = df.to_html()
    # --------------------------------------- Create and Send out HTML ------------------------------------------------- #

    html_string = '''
    <html>
        <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <!--<style>body{ margin:0 100; background:whitesmoke; }</style>-->
            <style>body{ margin:0 100;}</style>
        </head>
        <body>
            <h1>Daily Market Signals</h1>
    
            <!-- *** Section 1 *** --->
            <h2>The other solution is cloud based, make_subplotsmore interactive</h2>
            <h2>This in my opinion is more suitable for more static, like daily updates</h2>
            <div>
                ''' + div_table1 + '''
            </div>
            
            <h2>...more...</h2>
            <div>
            </div>
        </body>
    </html>'''

    return html_string
    #
    # with open('trading_opportunities.html', 'w') as f:
    #     f.write(html_string)