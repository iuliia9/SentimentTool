import pandas as pd
import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import sqlite3
import collections
import tweepy
from os import path
import re
import keras.backend.tensorflow_backend as tb

app = dash.Dash(__name__)
app.layout = html.Div([
html.H1("Sentiment Analysis", style={"textAlign": "center"}),
    # Dividing the dashboard in tabs
    dcc.Tabs(id="tabs", children=[
    # Defining the layout of the first Tab
        dcc.Tab(label='Quick Analysis', children=[
            html.Div([
                html.H1("Quick Static Sentiment",
                            style={'textAlign': 'center'}),
                dcc.Input(id='sentiment_term', value='Trump', type='text'),
                html.Div(['example'], id='input-div', style={'display': 'none'}),
                html.Button('Submit', id="submit-button"),
                html.Div(id='result'),
                dcc.Interval(
                id='res-update',
                interval=1 * 1000
               ),
                dcc.Graph(id='g3'),
                dcc.Graph(id='g4'),
                dcc.Interval(
                id='graph-update4',
                interval=1 * 1000
               )
                ]),
]),

# Defining the layout of the second Tab
        dcc.Tab(label='Live Anaysis', children=[
                html.H1("Live Sentiment",
                            style={"textAlign": "center"}),
                dcc.Input(id='sentiment_term2', value='Trump', type='text'),
                html.Div(['example'], id='input-div2', style={'display': 'none'}),
                html.Button('Submit', id="submit-button2"),
                html.Div(id='result2'),
                dcc.Interval(
                id='res-update2',
                interval=1 * 1000
               ),
                dcc.Graph(id='live-graph', animate=True),
                dcc.Interval(
                id='graph-update',
                interval=1 * 1000
                 ),
                 dcc.Graph(id='g2'),
                dcc.Interval(
                id='graph-update2',
                interval=1 * 1000
               ),
        ])
        ])
    ])

# average sentiment score for the static charts
@app.callback(
    Output('result', 'children'),
    [Input('res-update', 'n_intervals')]
)
def update_result(n):
    if (path.exists("scored.csv")):
        df = pd.read_csv("scored.csv")
        total = df["sentiment"].sum()
        rows = len(df.index)
        res = total/rows
        return "Average Sentiment Score is: {}".format(res)
    else:
        return "No Score to Display"



@app.callback(Output('input-div', 'children'),
              [Input('submit-button', 'n_clicks')],
              state=[State(component_id='sentiment_term', component_property='value')])

def update_div(n_clicks, input_value):
    return input_value


@app.callback(Output('input-div2', 'children'),
              [Input('submit-button2', 'n_clicks')],
              state=[State(component_id='sentiment_term2', component_property='value')])

def update_div(n_clicks, input_value):
    return input_value

# average sentiment score for the life graph
@app.callback(
    Output('result2', 'children'),
    [Input('res-update2', 'n_intervals'),
    Input('input-div2', 'children')]
)
def update_result(n, sentiment_term2):
        conn = sqlite3.connect('twitter_sentiment.db')
        c = conn.cursor()
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 2000",
                         conn, params=('%' + sentiment_term2 + '%',))
        total = sum(df.sentiment.values[-100:])
        # number of tweets in case less than 100
        num = len((df.sentiment.values[-100:]))
        res = total/num
        return "Average Sentiment Score is: {:.1f}".format(res)

@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals'),
               Input('input-div2', 'children')])

def update_graph_scatter(n, sentiment_term2):
    try:
        conn = sqlite3.connect('twitter_sentiment.db')
        c = conn.cursor()
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 2000",
                         conn ,params=('%' + sentiment_term2 + '%',))
        df.sort_values('unix', inplace=True)
        df['date'] = pd.to_datetime(df['unix'], unit='ms')
        df.set_index('date', inplace=True)
        df.dropna(inplace=True)
        X = df.index[-100:]
        Y = df.sentiment.values[-100:]
        if (df.empty == False):
            data = plotly.graph_objs.Scatter(
            x=X,
            y=Y,
            name='Scatter',
            text = df.tweet.values[-100:],
            mode='markers',
            opacity=0.8,
            marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            )
            return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]),
                                                    yaxis=dict(range=[min(Y)-0.1, max(Y)+0.1]),
                                                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                                        legend={'x': 0, 'y': 1},
                                                            hovermode='closest' )}
        else:
            return {'data': []}


    except Exception as e:
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')


@app.callback(Output('g2', 'figure'),
              [Input('graph-update2', 'n_intervals'),
               Input('input-div2', 'children')])

def update_pie_chart(n, sentiment_term2):
    conn = sqlite3.connect('twitter_sentiment.db')
    c = conn.cursor()
    df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 2000",
                     conn, params=('%' + sentiment_term2 + '%',))
    df.sort_values('unix', inplace=True)
    df['date'] = pd.to_datetime(df['unix'], unit='ms')
    df.set_index('date', inplace=True)
    df.dropna(inplace=True)
    Y = df.sentiment.values[-100:]
    labels = ['Negative', 'Neutral', 'Positive',]
    colors = ['#FF535D', '#C9CBCB', '#90FA75',]
    counter = collections.Counter(Y)
    values = [counter[0.0], counter[1.0], counter[2.0]]
    if (df.empty == False):
        trace = go.Pie(labels=labels, values=values,
                       hoverinfo='label+percent', textinfo='value',
                       textfont=dict(size=20),
                       marker=dict(colors=colors,
                                   line=dict(color='#000000', width=2)))
        return {'data': [trace], 'layout': go.Layout(title = 'Pie Chart'
                                                    )}
    else:
        return {'data': []}

#
@app.callback(Output('g3', 'figure'),
               [Input('input-div', 'children')])

def update_static_pie_chart(sentiment_term):
    pred = static_analysis(sentiment_term)
    labels = ['Negative', 'Neutral', 'Positive']
    colors = ['#FF535D', '#C9CBCB', '#90FA75']
    counter = collections.Counter(pred)
    values = [counter[0.0], counter[1.0], counter[2.0]]
    trace = go.Pie(labels=labels, values=values,
                   hoverinfo='label+percent', textinfo='value',
                   textfont=dict(size=20),
                   marker=dict(colors=colors,
                               line=dict(color='#000000', width=2)))
    return {'data': [trace], 'layout': go.Layout(title = 'Static Pie Chart'
                                                )}


@app.callback(Output('g4', 'figure'),
               [Input('graph-update4', 'n_intervals')])

def update_graph_scatter(sentiment_term):
    if (path.exists("scored.csv")):
        df = pd.read_csv("scored.csv")
        X = df.time
        Y = df.sentiment
        if (df.empty == False):
            data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                text=df.text,
                mode='markers',
                opacity=0.8,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                }
            )
            return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]),
                                                        yaxis=dict(range=[min(Y) - 0.1, max(Y) + 0.1]),
                                                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                                        legend={'x': 0, 'y': 1},
                                                        hovermode='closest')}
        else:
            return {'data': []}
    else:
        return {'data': []}

def preprocess_text(sen):
    sentence = re.sub("@[\w]*", '', sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def static_analysis(sentiment):
    tb._SYMBOLIC_SCOPE.value = True
    # load saved model
    file = open("model.json", 'r')
    model_json = file.read()
    file.close()
    from keras.models import model_from_json
    import pickle
    loaded_model = model_from_json(model_json)
    # load weights
    loaded_model.load_weights("model_weights.hdf5")
    # load tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    # Twitter API tokens
    auth = tweepy.OAuthHandler('consumer_key', 'consumer_secret')
    auth.set_access_token('access_token_key',
                          'access_token_secret')
    api = tweepy.API(auth)
    search_word = sentiment
    # create new dataframe
    df_tweets = pd.DataFrame(columns=['text'])
    # get tweets
    tweet_list = [];
    tweet_time = [];
    for tweet_info in tweepy.Cursor(api.search, q=search_word, lang="en",
                                    tweet_mode="extended").items(100):
        if "retweeted_status" in dir(tweet_info):
            tweet = tweet_info.retweeted_status.full_text
            tweet_list.append(tweet)
            tweet_time.append(tweet_info.created_at)
        else:
            tweet = tweet_info.full_text
            tweet_list.append(tweet)
            tweet_time.append(tweet_info.created_at)
    df_tweets["text"] = tweet_list
    df_tweets["time"] = tweet_time
    df_tweets.to_csv('result.csv')
    # preprocessing
    df_tweets['text'] = df_tweets['text'].apply(lambda x: x.lower())
    X = []
    sentences = list(df_tweets['text'])
    for sen in sentences:
        X.append(preprocess_text(sen))
    # tokenize and add padding
    sequences = loaded_tokenizer.texts_to_sequences(X)
    from keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences(sequences, padding='post', maxlen=20)
    # use model to get predictions
    pred = loaded_model.predict_classes(padded)
    # add sentiment scores to the dataframe
    df_tweets['sentiment'] = pred
    df_tweets.to_csv('scored.csv')
    return pred

if __name__ == '__main__':
    app.run_server(debug=True)



