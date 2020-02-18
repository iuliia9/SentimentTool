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
from keras.layers import Dense, Embedding, GRU, LSTM, SpatialDropout1D

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
print(tweets[ tweets['label'] == 'positive'].size)
print(tweets[ tweets['label'] == 'negative'].size)
print(tweets[ tweets['label'] == 'neutral'].size)

# preprocessing
tweets['text'] = tweets['text'].apply(lambda x: x.lower())
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
sentences = list(tweets['text'])
for sen in sentences:
    X.append(preprocess_text(sen))

y = tweets['label']
# convert labels to numbers
y = np.array(list(map(lambda x: 1 if x=="positive" else 0 if x=="negative" else 2 if x=="neutral" else 3, y)))

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
# Convert labels
labels = to_categorical(np.asarray(y))
X_train, X_test, y_train, y_test = train_test_split(padded_X, labels, test_size = 0.2, random_state = 0)
# Size of train and test datasets
print('X_train size:', X_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', X_test.shape)
print('y_test size:', y_test.shape)

# create embeddings
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix],
                           input_length = tweet_num, trainable=False)

# create our model
lstm_mod1 = Sequential()
lstm_mod1.add(embedding_layer)
lstm_mod1.add(LSTM(30,
            dropout = 0.5,
            recurrent_dropout = 0.5))
                # return_sequences = True))
# lstm_mod1.add(LSTM(128,
#             dropout = 0.5,
#             recurrent_dropout = 0.5))
lstm_mod1.add(Dense(3, activation='softmax'))
lstm_mod1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# lstm_mod1.summary()


hist_1 = lstm_mod1.fit(X_train, y_train,
                    validation_split = 0.2,
                    epochs=200, batch_size=256)

# train and test accuracy
loss, accuracy = lstm_mod1.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = lstm_mod1.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
# Plot train/test loss and accuracy
acc = hist_1.history['acc']
val_acc = hist_1.history['val_acc']
loss = hist_1.history['loss']
val_loss = hist_1.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# predicted values
y_pred = lstm_mod1.predict(X_test)
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
class_names = np.array(['negative', 'positive', 'neutral'])
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
    cm = confusion_matrix(y_true, y_pred)
    # labels from data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
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