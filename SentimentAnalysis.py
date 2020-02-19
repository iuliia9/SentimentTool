from keras.models import model_from_json
import pickle
import pandas as pd
import tweepy
from keras.preprocessing.sequence import pad_sequences
# load json and create model
file = open("model.json", 'r')
model_json = file.read()
file.close()

loaded_model = model_from_json(model_json)
# load weights
loaded_model.load_weights("model_weights.hdf5")
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)


auth = tweepy.OAuthHandler('consumer_key', 'consumer_secret')
auth.set_access_token('access_key', 'access_secret')
api = tweepy.API(auth)
search_word = input("Enter a search keyword:")
# create new dataframe
df_tweets = pd.DataFrame(columns = ['text'])
# get tweets
tweets = tweepy.Cursor(api.search,
                            q = search_word,
                            lang = "en").items(50)
tweet_list = [tweet for tweet in tweets]
# get text of tweets
for tweet in tweet_list:
    text = tweet.text
    add_tweet = [text]
    df_tweets.loc[len(df_tweets)] = add_tweet

# print(df_tweets['text'])
# save dataframe to csv
df_tweets.to_csv('result.csv')

# tokenize and add padding
sequences= loaded_tokenizer.texts_to_sequences(df_tweets['text'])
padded = pad_sequences(sequences, padding='post', maxlen=20)

# use model to get predictions
pred = loaded_model.predict_classes(padded)
# print all predicted values
print(pred)
# print predictions after tweets
for i in range(50):
	print((df_tweets['text'])[i], pred[i])
