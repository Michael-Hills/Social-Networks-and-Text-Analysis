from preprocessing import *
from positivity import *
from clusterQuery import *
import os
from sentence_transformers import SentenceTransformer
from bayesian import *
import pandas as pd


def loadData():
    """Function to load and process the data"""

    getFirstAndResponse()


def positivityRankings():
    """Function to run the positivity sentiment analysis functions"""

    #load the data
    loadData()
    
    #if the rankings have been calculated beforee, load from csv
    if(os.path.isfile(os.getcwd() + "/positivityScores.csv")):
        df = pd.read_csv('positivityScores.csv')
        print("Positivity scores loaded from csv")
    
    #if not, calculate them
    else:
        df = getScores()

    #find the highest and lowest average scores per company
    groups = orderByPositivity(df)

    #plot the scores
    plotScores(df,groups)

    #order the tweet scores in a company
    rankPositivity(df,groups)



def clusterQueries(search=False,company=0,rank=False):
    """Function to run the query clustering functions
    ---parameters---

    search: boolean, wether to perform bayesian optimisation or not
    company: int, to select the company to cluster from the top3 list"""


    #load the data
    loadData()

    #companies to analyse
    top3 = ['AppleSupport','AmazonHelp','Uber_Support']


    #if the clustering has been done before, load from jsonlines
    if(os.path.isfile(os.getcwd() + "/clustered"+ top3[company]+".jsonl")):
        df = pd.read_json("clustered"+ top3[company]+".jsonl",lines=True)
        print("Clusters loaded from jsonlines")
    
    #if not, create the sentence embessings and cluster them
    else:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        df = getSentenceVectors(model,top3)

        #bayesian search
        if search == True:
            getBayesianScores(df,top3[company])
        
        #no search, using hard coded parameters
        else:

            #get tweets to the company entered in the function parameter
            df = df[df['author_id_y'] == top3[company]]
            vectors = df['textVector'].values.tolist()

            #create the clusters
            clusterer = generateClusters(vectors,14,0.14,5,35,1)

            #save the cluster lables to a jsonlines file
            df['clusterLabel'] = clusterer.labels_

            #sort the dataframe by cluster label before writing to file
            df = df.sort_values('clusterLabel')
            df2 = pd.concat([df[df['clusterLabel'] != -1], df[df['clusterLabel'] == -1]])
            df2.to_json("clustered"+ top3[company]+".jsonl",orient='records', lines = True)



    #cluster to read
    cluster = 142

    #if true, print messages sorted by sentiment score
    if rank == True:
        rankInCluster(df,cluster)

    #else just print all messages in a cluster
    else:
        readCluster(df,cluster)

    TFIDF(df)

         

if __name__ == '__main__':

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    #Menu for options to run
    print("Menu:")
    print("1. Query sentiment analysis to rank the positivity/negativitiy of customer tweets")
    print("2. Cluster customer tweets into categories")

    while True:

        choice = input("Enter your choice (1/2): ")

        if choice == '1':
            positivityRankings()
            break
        elif choice == '2':
            clusterQueries(company=2,rank=True)
            break
        else:
            print("Invalid choice. Please select 1 or 2.")

    