import os
import pandas as pd
from tqdm import tqdm
from cluster import *
import jsonlines


def getSentenceVectors(model,groups):

    #return dataframe with the vectors if previously calculated
    if(os.path.isfile(os.getcwd() + "/queryVectors.jsonl")):
        df = pd.read_json('queryVectors.jsonl',lines=True)
        print("Vectors loaded from jsonlines file")
        return df

    #otherwise calculate the vectors
    if True:

        vectors = []
        df = pd.read_csv('firstResponse.csv')
        df = df[['tweet_id_x','author_id_x','author_id_y','text_x']]
        

        #calcuate only for the company selected
        mask = df['author_id_y'].isin(groups)
        df = df[mask]

               
        texts = df['text_x'].values.tolist()

        copy = df.copy()


        #encode the text to vectors
        for text in tqdm(texts):
            vectors.append(model.encode(text))
        

        #write vectors to a jsonlines file
        copy['textVector'] = vectors
        copy.dropna(subset=['textVector'],inplace=True)
        copy.to_json('queryVectors.jsonl',orient='records', lines = True)
       

        print("Vectors created and written to jsonlines file")
        return copy
    

