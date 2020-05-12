Sentiment Tool analyses the sentiment of tweets mentioning a certain keyword.
It uses LSTM neural network to perform sentiment analysis.
To run the tool, run SentimentAnalysis.py, wait for a few seconds (until you see tweets displayed in you rterminal), then run test.py. 
Click on the link that will appear in your terminal window after you start running test.py. This will take you to the dashboard. 

The neural network was built in BuildModel.py. If you run that file, you will be re-training the model (will take a while).
However, if you do wish to run the file, note the following:
Two datasets were used to train the model (in BuildModel.py):
1. corpus created by Nick Sanders (downloaded from https://github.com/karanluthra/twitter-sentiment-training/blob/master/corpus.csv)
2. Twitter US Airline Sentiment (downloaded from Kaggle https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
