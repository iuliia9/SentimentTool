import twitter

# initialize api instance
twitter_api = twitter.Api(consumer_key='token',
                        consumer_secret='token',
                        access_token_key='token',
                        access_token_secret='token')

# test authentication
print(twitter_api.VerifyCredentials())




def buidTrainingSet(corpusFile, tweetDataFile):
    import csv
    import time

    corpus = []

    with open(corpusFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id": row[2], "label": row[1], "topic": row[0]})

    rate_limit = 180
    sleep_time = 900 / 180

    trainingDataSet = []

    for tweet in corpus:
        try:
            status = twitter_api.GetStatus(tweet["tweet_id"])
            print("Tweet fetched" + status.text)
            tweet["text"] = status.text
            trainingDataSet.append(tweet)
            time.sleep(sleep_time)
        except:
            continue
    # now we write them to the empty CSV file
    with open(tweetDataFile, 'w') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=',', quotechar="\"")
        for tweet in trainingDataSet:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception as e:
                print(e)
    return trainingDataSet

corpusFile = "corpus.csv"
tweetDataFile = "tweetDataFile.csv"

trainingData = buidTrainingSet(corpusFile, tweetDataFile)
trainingData1 = trainingData