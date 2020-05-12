Sentiment Tool analyses the sentiment of tweets mentioning a certain keyword.
It uses LSTM neural network to perform the sentiment analysis and Dash (Python framework) to display the results and to receive the input keyword.
To run the tool, you need to run two files simultaneously. Run SentimentAnalysis.py, wait for a few seconds (until you see tweets displayed in your terminal), then run test.py. 
Click on the link that will appear in your terminal window after you start running test.py. This will take you to the dashboard.

This project was built with Python 3.7 and tensorflow. 

The neural network was built in BuildModel.py. If you run that file, you will be re-training the model (will take a while).
However, if you do wish to run the file, note the following:
Two datasets were used to train the model (in BuildModel.py):
1. corpus created by Nick Sanders (downloaded from https://github.com/karanluthra/twitter-sentiment-training/blob/master/corpus.csv)
2. Twitter US Airline Sentiment (downloaded from Kaggle https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

Other files: 
1. TwitterAPI.py was used to download the tweets by tweet ids (tweets from Nick Sanders corpus). This file needs to be run if you wish to download the dataset with tweets. Note that running this file will take a very long time. 
2. UI.py - this file contains the code for the first simple tkinter UI created for the project. Even though this UI is not used in the final version of the tool, this file was left to show the progress of the project. 
3. Tokenizer, model, and model weights contain information about the model. Download these files so that you can run the tool. Without these files, it will be necessary to re-run BuildModel.py to create and save a new model. 
