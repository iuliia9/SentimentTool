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
                html.Button('Submit', id="submit_button"),
                html.Div(id='result'),
                dcc.Interval(
                id='resupdate',
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
    [Input('resupdate', 'n_intervals')]
)
def update_result(resupdate):
    if (path.exists("scored.csv")):
        df = pd.read_csv("scored.csv")
        total = df["sentiment"].sum()
        rows = len(df.index)
        if rows != 0:
            res = total / rows
            return "Average Sentiment Score is: {}".format(res)
        else:
            return "No Score to Display"
    else:
        return "No Score to Display"


# on click of submit button static tab
@app.callback(Output('input-div', 'children'),
              [Input('submit_button', 'n_clicks')],
              state=[State(component_id='sentiment_term', component_property='value')])

def update_div(n_clicks, sentiment_term):
    return sentiment_term

# on click of submit button live tab
@app.callback(Output('input-div2', 'children'),
              [Input('submit-button2', 'n_clicks')],
              state=[State(component_id='sentiment_term2', component_property='value')])

def update_div(n_clicks, sentiment_term2):
    return sentiment_term2

# average sentiment score for the live graph
@app.callback(
    Output('result2', 'children'),
    [Input('res-update2', 'n_intervals'),
    Input('input-div2', 'children')]
)
def update_result(n_intervals, sentiment_term2):
        conn = sqlite3.connect('twitter_sentiment.db')
        c = conn.cursor()
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 150",
                         conn, params=('%' + sentiment_term2 + '%',))
        total = sum(df.sentiment.values[-100:])
        # number of tweets in case less than 100
        num = len((df.sentiment.values[-100:]))
        # if there are tweets matching the keyword
        if num != 0:
            res = total/num
            return "Average Sentiment Score is: {:.1f}".format(res)
        # if there are no tweets matching the keyword
        else:
            return "No Score to Display"

# update live scatter plot
@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals'),
               Input('input-div2', 'children')])

def update_graph_scatter(n, sentiment_term2):
        conn = sqlite3.connect('twitter_sentiment.db')
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 150",
                         conn ,params=('%' + sentiment_term2 + '%',))
        df.sort_values('unix', inplace=True)
        # convert date and time of each tweet to datetime
        from datetime import datetime
        # get a list of all date and time
        tweetsTime = list(df['unix'])
        # create an empty list where converted items will go
        time = []
        # go through tweetsText, convert and add to time list
        for tweet in tweetsTime:
            tweet = datetime.strptime(tweet, '%Y-%m-%d %H:%M:%S')
            time.append(tweet)
        # create a new column where converted datetime will go
        df['date'] = time
        # get last 100 datetimes and sentiment labels to display
        X = df.date[-100:]
        Y = df.sentiment.values[-100:]
        # make sure the dataframe is not empty
        # the dataframe will be empty if there are no tweets matching the keyword in database
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

@app.callback(Output('g2', 'figure'),
              [Input('graph-update2', 'n_intervals'),
               Input('input-div2', 'children')])

def update_pie_chart(n, sentiment_term2):
    # connect to database
    conn = sqlite3.connect('twitter_sentiment.db')
    # read sql into a dataframe
    df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE ? ORDER BY unix DESC LIMIT 150",
                     conn, params=('%' + sentiment_term2 + '%',))
    # get the last 100 sentiment labels
    Y = df.sentiment.values[-100:]
    labels = ['Negative', 'Neutral', 'Positive',]
    colors = ['#FF535D', '#C9CBCB', '#90FA75',]
    # count how many negative, neutral and positive labels
    counter = collections.Counter(Y)
    values = [counter[0.0], counter[1.0], counter[2.0]]
    # make sure the dataframe is not empty
    # the dataframe will be empty if there are no tweets matching the keyword
    if (df.empty == False):
        trace = go.Pie(labels=labels, values=values,
                       hoverinfo='label+percent', textinfo='value',
                       textfont=dict(size=20),
                       marker=dict(colors=colors,
                                   line=dict(color='#000000', width=2)))
        return {'data': [trace], 'layout': go.Layout(title = 'Pie Chart'
                                                    )}
    # if datafrane is empty return empty graph
    else:
        return {'data': []}

# update static pie chart
@app.callback(Output('g3', 'figure'),
               [Input('input-div', 'children')])

def update_static_pie_chart(sentiment_term):
    # do sentiment analysis and get sentiment labels
    pred = static_analysis(sentiment_term)
    # define label names
    labels = ['Negative', 'Neutral', 'Positive']
    # define label colours
    colors = ['#FF535D', '#C9CBCB', '#90FA75']
    # count how many negative, neutral and positive labels there are
    counter = collections.Counter(pred)
    values = [counter[0.0], counter[1.0], counter[2.0]]
    # make sure that the predictions exist
    # the predictions will not exist if there were no tweets matching the keyword
    if pred != []:
        trace = go.Pie(labels=labels, values=values,
                       hoverinfo='label+percent', textinfo='value',
                       textfont=dict(size=20),
                       marker=dict(colors=colors,
                                   line=dict(color='#000000', width=2)))
        return {'data': [trace], 'layout': go.Layout(title = 'Static Pie Chart'
                                                   )}
    # if there are no predictions return empty graph
    else:
        return {'data': []}

# update static scatter chart
@app.callback(Output('g4', 'figure'),
               [Input('graph-update4', 'n_intervals')])

def update_graph_scatter(self):
    # make sure the csv file with results exists
    # the file will not exist when the tool is run for the very first time
    if (path.exists("scored.csv")):
        # create a dataframe
        df = pd.read_csv("scored.csv")
        # get sentimnt labels
        Y = df.sentiment
        # get tweets time
        X = df.time
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
        # make sure the dataframe is not empty
        # it will be empty if there are no tweets matching the keyword
        if (df.empty == False):
            return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]),
                                                                yaxis=dict(range=[min(Y) - 0.1, max(Y) + 0.1]),
                                                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                                                legend={'x': 0, 'y': 1},
                                                                hovermode='closest')}
        else:
            # if the search keyword returned no results - scored.csv is empty
            # display empty graph
            # one possible reason for no results - a spelling mistake in the search keyword
            # another possible reason - a rare keyword, so that it has not been mentioned on Twitter in the last 7 days
            return {'data': []}
    # if the tool is run for the first time and scored.csv has not been created yet display empty graph
    else:
        return {'data': []}

# text preprocessing before sentiment analysis
def preprocess_text(tweet):
    # define patterns

    # anything that is not alphabetic characters
    replace = re.compile('[^a-z]')
    # multiple spaces
    multipleSpaces = re.compile('\s+')
    # single letters, like 'a'
    singleLetter = re.compile(r"\b[a-z]\b")
    # replace matched patterns with a space
    tweet = replace.sub(' ', tweet)
    tweet = singleLetter.sub(' ', tweet)
    tweet = multipleSpaces.sub(' ', tweet)
    return tweet


# perform sentiment analysis
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
    # define a search keyword
    search_word = sentiment
    # create new dataframe
    df_tweets = pd.DataFrame(columns=['text', 'time', 'sentiment'])
    # create empty lists where tweets texts and time will be saved
    tweet_list = [];
    tweet_time = [];
    for tweet_info in tweepy.Cursor(api.search, q=search_word, lang="en",
                                    tweet_mode="extended").items(100):
        # if it is a retweet
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
    # preprocessing
    df_tweets['text'] = df_tweets['text'].apply(lambda x: x.lower())
    X = []
    tweetTexts = list(df_tweets['text'])
    for tweet in tweetTexts:
        X.append(preprocess_text(tweet))
    # make sure the tweets exist
    # they will not exist when there are no tweets matching the keyword
    if X != []:
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
    else:
        df_tweets.to_csv('scored.csv')
        return []

if __name__ == '__main__':
    app.run_server(debug=True)



