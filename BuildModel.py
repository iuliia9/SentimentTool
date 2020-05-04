# tutorials were used to learn how to implement an lstm model
# https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
# https://medium.com/@francesca_lim/twitter-u-s-airline-sentiment-analysis-using-keras-and-rnns-1956f42294ef
import re
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM
import h5py

# create column names for dataframes
colnames=['id', 'text', 'label', 'company']
aircolnames=['tweet_id','label','airline_sentiment_confidence','negativereason',
'negativereason_confidence','airline', 'airline_sentiment_gold','name','negativereason_gold',
'retweet_count','text','tweet_coord','tweet_created','tweet_location','user_timezone']

# read from csv files
tweets = pd.read_csv("tweetDataFile.csv", names=colnames, header=None)
airtweets = pd.read_csv('Tweets.csv', names=aircolnames, header=None)

# leave only the columns we need
airtweets = airtweets.drop(airtweets.index[[0]])
tweets = tweets[['text','label']]
airtweets=airtweets[['text', 'label']]

# combine two dataframes
frames = [airtweets, tweets]
tweets = pd.concat(frames)

# remove irrelevant label as it is present only in one dataframe
tweets = tweets[tweets.label != "irrelevant"]

# print number of positive, negative and neutral tweets
# print(tweets[ tweets['label'] == 'positive'].size)
# print(tweets[ tweets['label'] == 'negative'].size)
# print(tweets[ tweets['label'] == 'neutral'].size)

# data preprocessing
# to lower characters
tweets['text'] = tweets['text'].apply(lambda x: x.lower())
def preprocess(tweet):
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

X = []
tweetsText = list(tweets['text'])
for tweet in tweetsText:
    X.append(preprocess(tweet))

# sentiment labels
y = tweets['label']
# convert text labels to numbers
y = np.array(list(map(lambda x: 2 if x=="positive" else 0 if x=="negative" else 1, y)))

# tokenizer
t = Tokenizer()
t.fit_on_texts(X)
# number of unique words
vocab_size = len(t.word_index) + 1
# integer encode
sequences = t.texts_to_sequences(X)
# longest tweet in sequences
def max_tweet_length():
    for i in range(1, len(sequences)):
        max_length = len(sequences[0])
        if len(sequences[i]) > max_length:
            max_length = len(sequences[i])
    return max_length
num_char = max_tweet_length()
maxlength = num_char

padded_X = pad_sequences(sequences, padding='post', maxlen=maxlength)

# Convert labels to categorical for loss function
labels = to_categorical(np.asarray(y))

# split data in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_X, labels, test_size = 0.2, random_state = 0)
# Size of train and test datasets
# print('X_train size:', X_train.shape)
# print('y_train size:', y_train.shape)
# print('X_test size:', X_test.shape)
# print('y_test size:', y_test.shape)

# create embeddings
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
# print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# create embedding layer
embedding_layer = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix],
                           input_length = num_char, trainable=False)

# create model
lstm_model = Sequential()
lstm_model.add(embedding_layer)
lstm_model.add(LSTM(30, dropout = 0.5, recurrent_dropout = 0.5))
lstm_model.add(Dense(3, activation='softmax'))
lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# introduce early early_stopping
# wait for 10 epochs before stopping in case the model improves
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# train model
hist = lstm_model.fit(X_train, y_train,
                      validation_split = 0.2,
                      epochs=200, batch_size=256, verbose=1, callbacks=[early_stopping])

# saving the model
# serialize model to JSON
json_file = lstm_model.to_json()
with open("model.json", "w") as file:
   file.write(json_file)
# # serialize model weights to HDF5
lstm_model.save_weights("model_weights.hdf5")
# save tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
     pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

# get training accuracy
loss, accuracy = lstm_model.evaluate(X_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
# test model and get test accuracy
loss, accuracy = lstm_model.evaluate(X_test, y_test, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# plot train vs test accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Training Accuracy vs. Testing Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training accuracy', 'Testing accuracy'], loc='lower right')
plt.show()
# plot train vs test loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training loss', 'Testing loss'], loc='upper right')
plt.show()


# create confusion matrix
# https://medium.com/@francesca_lim/twitter-u-s-airline-sentiment-analysis-using-keras-and-rnns-1956f42294ef
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# predicted values
y_pred = lstm_model.predict(X_test)
# empty numpy array same length as training so that predictions can be matched with training
y_pred_array = np.zeros(X_test.shape[0])
# class with highest probability
for i in range(0, y_pred.shape[0]):
    label_predict = np.argmax(y_pred[i])
    y_pred_array[i] = label_predict
# convert to integers
y_pred_array = y_pred_array.astype(int)
# Convert y_test to 1d numpy array
y_test_array = np.zeros(X_test.shape[0])
for i in range(0, y_test.shape[0]):
    label_predict = np.argmax(y_test[i])
    y_test_array[i] = label_predict
y_test_array = y_test_array.astype(int)
class_names = np.array(['negative', 'neutral', 'positive'])
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
# can be normalized
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # get confusion matrix
    cmatrix = confusion_matrix(y_true, y_pred)
    # labels from data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cmatrix = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cmatrix)

    fig, ax = plt.subplots()
    im = ax.imshow(cmatrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cmatrix.shape[1]),
           yticks=np.arange(cmatrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cmatrix.max() / 2.
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax.text(j, i, format(cmatrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cmatrix[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test_array, y_pred_array, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plot_confusion_matrix(y_test_array, y_pred_array, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
# print plot
plt.show()





