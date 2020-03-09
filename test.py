import pandas as pd
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import sqlite3

app = dash.Dash(__name__)
app.layout = html.Div(
    [   html.H2('Twitter Sentiment: Live'),
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),
    ]
)

@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_graph_scatter(self):
    try:
        conn = sqlite3.connect('twitter_sentiment.db')
        c = conn.cursor()
        df = pd.read_sql("SELECT * FROM sentiment ORDER BY unix DESC LIMIT 1000", conn)
        df.sort_values('unix', inplace=True)
        X = df.unix.values[-100:]
        Y = df.sentiment.values[-100:]
        data = plotly.graph_objs.Scatter(
        x=X,
        y=Y,
        name='Scatter',
        mode='markers'
        )

        return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]),
                                                yaxis=dict(range=[min(Y)-0.1, max(Y)+0.1]), )}

    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')

if __name__ == '__main__':
    app.run_server(debug=True)