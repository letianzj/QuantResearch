import os
import pandas as pd
import numpy as np

from IPython.display import display, HTML

# --------------------------------------------- Create contents ----------------------------------------------------- #
def generate_html():
    graphs = [
        'https://plotly.com/~christopherp/308',
        'https://plotly.com/~christopherp/306',
        'https://plotly.com/~christopherp/300',
        'https://plotly.com/~christopherp/296'
    ]

    template = (''
                '<a href="{graph_url}" target="_blank">'  # Open the interactive graph when you click on the image
                '<img src="{graph_url}.png">'  # Use the ".png" magic url so that the latest, most-up-to-date image is included
                '</a>'
                '{caption}'  # Optional caption to include below the graph
                '<br>'  # Line break
                '<a href="{graph_url}" style="color: rgb(190,190,190); text-decoration: none; font-weight: 200;" target="_blank">'
                'Click to comment and see the interactive graph'  # Direct readers to Plotly for commenting, interactive graph
                '</a>'
                '<br>'
                '<hr>'  # horizontal line
                '')

    email_body = ''
    for graph in graphs:
        _ = template
        _ = _.format(graph_url=graph, caption='')
        email_body += _

    # display(HTML(email_body))

    return email_body

# TODO: email out trading_opportunities.html
