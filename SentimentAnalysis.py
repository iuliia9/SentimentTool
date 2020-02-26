from keras.models import model_from_json
import pickle
import re
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
auth.set_access_token('access_token_key', 'access_token_secret')
api = tweepy.API(auth)
search_word = input("Enter a search keyword:")
# create new dataframe
df_tweets = pd.DataFrame(columns = ['text'])
# get tweets
tweet_list =[];
for tweet_info in tweepy.Cursor(api.search, q=search_word, lang = "en",
                                tweet_mode="extended").items(100):
    if "retweeted_status" in dir(tweet_info):
        tweet=tweet_info.retweeted_status.full_text
        tweet_list.append(tweet)
    else:
        tweet=tweet_info.full_text
        tweet_list.append(tweet)
# get text of tweets
for tweet in tweet_list:
    add_tweet = [tweet]
    df_tweets.loc[len(df_tweets)] = add_tweet

# save dataframe to csv
df_tweets.to_csv('result.csv')

# preprocessing
df_tweets['text'] = df_tweets['text'].apply(lambda x: x.lower())
def preprocess_text(sen):
    sentence = re.sub("@[\w]*", '', sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

X = []
sentences = list(df_tweets['text'])
for sen in sentences:
    X.append(preprocess_text(sen))
# tokenize and add padding
sequences= loaded_tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, padding='post', maxlen=20)

# use model to get predictions
pred = loaded_model.predict_classes(padded)
# print all predicted values
print(pred)
# print predictions after tweets
for i in range(100):
	print((df_tweets['text'])[i], pred[i])
