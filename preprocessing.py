import pandas as pd
import re
import os


def getFirstAndResponse():
    """
    Function to get all opening messages to a company and their replies
    """

    #if messages found before, read the csv
    if(os.path.isfile(os.getcwd() + "/firstResponse.csv")):
        
        print("First message dataset loaded")
    
    #else, read the original dataset
    else:
    
        tweets = pd.read_csv('twcs.csv')

        #find all tweets that are the first in a conversation
        first_inbound = tweets[pd.isnull(tweets.in_response_to_tweet_id) & tweets.inbound]
        print('Found {} first inbound messages.'.format(len(first_inbound)))

    

        #find the first reply to those tweets
        inbounds_and_outbounds = pd.merge(first_inbound, tweets, left_on='tweet_id', 
                                        right_on='in_response_to_tweet_id')


        #remove any where the reply isnt from the company
        inbounds_and_outbounds = inbounds_and_outbounds[inbounds_and_outbounds.inbound_y ^ True]

        print("Found {} responses.".format(len(inbounds_and_outbounds)))

        pd.DataFrame.to_csv(inbounds_and_outbounds,'firstResponse.csv')

        print("File read, processed and stored")
               






