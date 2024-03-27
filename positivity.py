from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def getDataframe(fileName):
    """Function to read a csv file, selecting only certain columns"""

    df = pd.read_csv(fileName,usecols=['author_id_x','text_x','author_id_y'])
    return df


def groupByAuthor(df):
    """Function to group all tweets by author"""
    
    grouped = df.groupby('author_id_y').size().sort_values(ascending=False)
    return grouped


def getScores():
    """Function to get the positivity score calculated with VADER sentiment analysis"""


    #Load the sentiment analysis tool
    sid_obj = SentimentIntensityAnalyzer()
    
    #read the csv and extract the tweet text
    scores = []
    df = pd.read_csv('firstResponse.csv')
    textdf = df[['author_id_x','text_x','author_id_y']]
    texts = textdf['text_x'].values.tolist()

    copy = textdf.copy()

    #calculate positivity for all tweets
    for text in tqdm(texts):
        sentiment_dict = sid_obj.polarity_scores(text)
        scores.append(sentiment_dict['compound'])



    #write the positivity scores to a csv
    copy['positivity'] = scores
    copy.dropna(subset=['positivity'],inplace=True)
    copy.to_csv('positivityScores.csv')

    return copy


def plotScores(df,groups):

    """Function to plot the positivity score for a group of companies"""

    plt.figure(figsize=(8, 6))
    
    #plot the scores of all messages within the author group
    for group in groups:
        groupDf = df[df['author_id_y'] == group]
        positivityScores = groupDf['positivity'].values.tolist()
        sns.kdeplot(positivityScores, label=group)
        plt.legend(title='Support team')

    plt.title('Kernel Density Estimation (KDE) Plot of Positivity Scores')
    plt.xlabel('Positivity Score')
    plt.ylabel('Density')
    plt.show()

def orderByPositivity(df):
    """Function to get the 2 highest and lowest average scores"""

    ordered = df.groupby("author_id_y", as_index=False).positivity.mean().sort_values('positivity', ascending=False)
    print(ordered.to_string())

    #get top 2 and bottom 2 by average score
    groups = ordered.iloc[[0,1,-2,-1]]
    return groups['author_id_y'].values.tolist()

    
def rankPositivity(df,groups):
    """Function to sort the tweets to a single company lowest to highest"""

    for group in groups:
        groupDf = df[df['author_id_y'] == group]
        sortedDf = groupDf.sort_values(by=['positivity'], ascending=True)
        print(sortedDf)

def rankInCluster(df,label):

    #get all messages within a cluster
    df = df[df['clusterLabel'] == label]
    scores = []
    texts = df['text_x'].values.tolist()

    sid_obj = SentimentIntensityAnalyzer()

    #calculate each messages positivity
    for text in texts:
        sentiment_dict = sid_obj.polarity_scores(text)
        scores.append(sentiment_dict['compound'])

    copy = df.copy()

    #write the positivity scores to a csv
    copy['positivity'] = scores
    copy.dropna(subset=['positivity'],inplace=True)
    sortedDf = copy.sort_values(by=['positivity'], ascending=True)
    print(sortedDf['text_x'].values.tolist())


   

    
