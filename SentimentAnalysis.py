from keras.models import model_from_json
import pickle
import matplotlib.pyplot as plt
import re
import pandas as pd
import tweepy
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import collections
from keras.preprocessing.text import text_to_word_sequence
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sqlite3
from unidecode import unidecode
import time

# def main(keyword):
# load json and create model
file = open("model.json", 'r')
model_json = file.read()
file.close()

loaded_model = model_from_json(model_json)
# load weights
loaded_model.load_weights("model_weights.hdf5")
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

conn = sqlite3.connect('twitter_sentiment.db')
c = conn.cursor()
def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, sentiment REAL)")
    conn.commit()

    create_table()

auth = tweepy.OAuthHandler('consumer_key', 'consumer_secret')
auth.set_access_token('access_token_key', 'access_token_secret')

conn = sqlite3.connect('twitter_sentiment.db')
c = conn.cursor()
def create_table():
    try:
        c.execute("CREATE TABLE IF NOT EXISTS sentiment(unix REAL, tweet TEXT, sentiment REAL)")
        c.execute("CREATE INDEX fast_unix ON sentiment(unix)")
        c.execute("CREATE INDEX fast_tweet ON sentiment(tweet)")
        c.execute("CREATE INDEX fast_sentiment ON sentiment(sentiment)")
        conn.commit()
    except Exception as e:
        print(str(e))
create_table()

class listener(StreamListener):

    def on_data(self, data):
        try:
            data = json.loads(data)
            tweet = data['text']
            time_ms = data['timestamp_ms']
            # preprocessing
            # tweet = tweet.apply(lambda x: x.lower())
            tweet = re.sub("@[\w]*", '', tweet)
            # Remove punctuations and numbers
            tweet = re.sub('[^a-zA-Z]', ' ', tweet)
            # Single character removal
            tweet = re.sub(r"\s+[a-zA-Z]\s+", ' ', tweet)
            # Removing multiple spaces
            tweet = re.sub(r'\s+', ' ', tweet)

            sequences= loaded_tokenizer.texts_to_sequences([tweet])
            padded = pad_sequences(sequences, padding='post', maxlen=20)
            sentiment = loaded_model.predict_classes(padded)[0]
            print(time_ms, tweet, sentiment)
            c.execute("INSERT INTO sentiment (unix, tweet, sentiment) VALUES (?, ?, ?)",
                      (time_ms, tweet, sentiment))
            conn.commit()

        except KeyError as e:
            print(str(e))
        return (True)

    def on_error(self, status):
            print(status)
while True:

    try:
        twitterStream = Stream(auth, listener())
        twitterStream.filter(track=["Trump"])
    except Exception as e:
        print(str(e))
        time.sleep(5)


#     api = tweepy.API(auth)
#     search_word = keyword
#     # search_word = input("Enter a search keyword:")
#     # create new dataframe
#     df_tweets = pd.DataFrame(columns = ['text'])
#     # get tweets
#     tweet_list =[];
#     tweet_time=[];
#     for tweet_info in tweepy.Cursor(api.search, q=search_word, lang = "en",
#                                     tweet_mode="extended").items(100):
#         if "retweeted_status" in dir(tweet_info):
#             tweet=tweet_info.retweeted_status.full_text
#             tweet_list.append(tweet)
#             tweet_time.append(tweet_info.created_at)
#         else:
#             tweet=tweet_info.full_text
#             tweet_list.append(tweet)
#             tweet_time.append(tweet_info.created_at)
#     # get text of tweets
#     for tweet in tweet_list:
#         add_tweet = [tweet]
#         df_tweets.loc[len(df_tweets)] = add_tweet
#     print(tweet_time)
#     # save dataframe to csv
#     df_tweets.to_csv('result.csv')
#
#     # preprocessing
#     df_tweets['text'] = df_tweets['text'].apply(lambda x: x.lower())
#
#     X = []
#     sentences = list(df_tweets['text'])
#     for sen in sentences:
#         X.append(preprocess_text(sen))
#     # tokenize and add padding
#     sequences= loaded_tokenizer.texts_to_sequences(X)
#     padded = pad_sequences(sequences, padding='post', maxlen=20)
#
#     # use model to get predictions
#     pred = loaded_model.predict_classes(padded)
#     # print all predicted values
#     print(pred)
#     # add sentiment scores to the dataframe
#     df_tweets['sentiment']=pred
#     df_tweets.to_csv('scored.csv')
#     # print predictions after tweets
#     # for i in range(100):
#     #     print((df_tweets['text'])[i], pred[i])
#     counter = collections.Counter(pred)
#     print(counter)
#     plot_bar_chart(counter)
#     # print(counter[0])
#     # print(counter[1])
#     # print(counter[2])
#
#
#
# def plot_bar_chart(counter):
#     label=['Negative', 'Positive', 'Neutral']
#     sentiment = [counter[0], counter[1], counter[2]]
#     index = np.arange(len(label))
#     plt.bar(index, sentiment, color=['red', 'green', 'blue'])
#     plt.xlabel('Sentiment', fontsize=5)
#     plt.ylabel('Number of tweets', fontsize=5)
#     plt.xticks(index, label, fontsize=5, rotation=30)
#     plt.title('Sentiment Analysis')
#     plt.show()
#
#
# def preprocess_text(sen):
#     sentence = re.sub("@[\w]*", '', sen)
#     # Remove punctuations and numbers
#     sentence = re.sub('[^a-zA-Z]', ' ', sentence)
#     # Single character removal
#     sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
#     # Removing multiple spaces
#     sentence = re.sub(r'\s+', ' ', sentence)
#     return sentence
#
 # main()