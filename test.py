import pandas as pd
import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import sqlite3



app = dash.Dash(__name__)
app.layout = html.Div(
    [   html.H2('Twitter Sentiment: Live'),
        dcc.Input(id='sentiment_term', value='Trump', type='text'),
        html.Div(['example'], id='input-div', style={'display': 'none'}),
        html.Button('Submit', id="submit-button"),
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),
    ]
)
@app.callback(Output('input-div', 'children'),
              [Input('submit-button', 'n_clicks')],
              state=[State(component_id='sentiment_term', component_property='value')])
def update_div(n_clicks, input_value):
    return input_value

@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals'),
               Input('input-div', 'children')])

# @app.callback(Output('live-graph', 'figure'),
#               [Input('graph-update', 'n_intervals'),
#                Input(component_id='sentiment_term', component_property='value')])
def update_graph_scatter(n, sentiment_term):
    try:
        conn = sqlite3.connect('twitter_sentiment.db')
        c = conn.cursor()
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 2000",
                         conn ,params=('%' + sentiment_term + '%',))

        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        df.dropna(inplace=True)
        X = df.index[-100:]
            # df.unix.values[-200:]
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