from keras.models import model_from_json
import pickle
import re
import tweepy
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tweepy import Stream
from tweepy.streaming import StreamListener
import json
import sqlite3
import time

sqlite3.register_adapter(np.int64, int)
sqlite3.register_adapter(np.int32, int)


# load saved json model
file = open("model.json", 'r')
model_json = file.read()
file.close()
loaded_model = model_from_json(model_json)
# load weights
loaded_model.load_weights("model_weights.hdf5")
# load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Twitter API tokens
auth = tweepy.OAuthHandler('consumer_key', 'consumer_secret')
auth.set_access_token('access_token_key', 'access_token_secret')

# connect to the database
conn = sqlite3.connect('twitter_sentiment.db')
c = conn.cursor()
# create a new table in the database if it does not already exist
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

# stream tweets
class listener(StreamListener):
    def on_data(self, data):
        try:
            data = json.loads(data)
            tweet = data['text']
            time_ms = data['timestamp_ms']
            # data preprocessing
            tweet = re.sub("@[\w]*", '', tweet)
            # Remove punctuations and numbers
            tweet = re.sub('[^a-zA-Z]', ' ', tweet)
            # Single character removal
            tweet = re.sub(r"\s+[a-zA-Z]\s+", ' ', tweet)
            # Removing multiple spaces
            tweet = re.sub(r'\s+', ' ', tweet)

            sequences= loaded_tokenizer.texts_to_sequences([tweet])
            padded = pad_sequences(sequences, padding='post', maxlen=20)
            # use model to predict the sentiment
            sentiment = loaded_model.predict_classes(padded)[0]
            # print the tweet
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
        # listen to all tweets in English (that have vowels)
        twitterStream = Stream(auth, listener(), tweet_mode = "extended")
        twitterStream.filter(languages=["en"], track=["a","e","i","o","u"])
    except Exception as e:
        print(str(e))
        time.sleep(5)