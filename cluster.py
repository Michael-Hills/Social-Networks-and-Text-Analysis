import umap.umap_ as umap
import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from bayesian import scoreClusters,scoreClustersWithoutNoise
import nltk
import re



def generateClusters(data,n_neighbors,min_dist,n_components,min_cluster_size,min_samples):
    """
    Function to create clusters
    """

    #reduce dimensions with UMAP
    clusterFit = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric='cosine'
        )
    
    
    reduced = clusterFit.fit_transform(data)

    #cluster with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,gen_min_span_tree=True).fit(reduced)

    #print both cluster costs
    print(scoreClusters(clusterer,0.2))
    print(scoreClustersWithoutNoise(clusterer,0.2))

    return clusterer
            
    
def visualiseClusters(data,n_neighbors,min_dist):
    """
    Function to show the clusters in 2D space
    """

    labels = data['clusterLabel'].values.tolist()
    
    visualisationFit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=83,
            metric='cosine'
        )

    visualise = visualisationFit.fit_transform(data)
                    
    fig2 = px.scatter(visualise, x=0, y=1, color = labels)
    fig2.update_traces(mode="markers")
    fig2.show()
                
    plt.show()


def TFIDF(df):
    """
    Function to perform TF-IDF to label clusters
    """

    noClusters = df['clusterLabel'].max() + 1

    #load the tf-idf vectoriser
    tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english')

    clusterTexts = []

    print("Concatenating texts for TF-IDF...")
    for i in range(noClusters):
       
        #read the file containing the text clusters
        tweets = df.loc[df['clusterLabel'] == i]
        tweets = tweets['text_x'].values.tolist()
        tweets = ' '.join(tweets)


        #remove strings starting with @
        pattern = r'\b@\w+\b'
        tweets = re.sub(pattern, '', tweets)

        #retrieve stopwrds and common words
        stopwords = nltk.corpus.stopwords.words('english')
        additional_stopwords = ['apple', 'amazon', 'uber', 'support', 'care','uber_support','applesupport','http','https','help','amazonhelp']
        stopwordsList = set(stopwords + additional_stopwords)
        lemmatizer = WordNetLemmatizer()
        

        processed_text = re.sub('[^a-zA-Z]', ' ',tweets)   #remove numbers
        processed_text = processed_text.lower()            #set to lowercase
        processed_text = processed_text.split()         

        #lemmatize words 
        processed_text = [lemmatizer.lemmatize(word, pos='a') for word in processed_text if word not in set(stopwordsList)]
        processed_text = [lemmatizer.lemmatize(word, pos='v') for word in processed_text if word not in set(stopwordsList)]
        processed_text = [lemmatizer.lemmatize(word, pos='n') for word in processed_text if word not in set(stopwordsList)]
        processed_text = ' '.join(processed_text)


        clusterTexts.append(processed_text)
   

    labels = []

    #get the top 5 words using TF-IDF per cluster
    tfidf_vector = tfidf_vectorizer.fit_transform(clusterTexts)
    tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    transposed = tfidf_df.T

    print("Calculating TF-IDF...")

    for i in range(noClusters):
        print(i, transposed.sort_values(by=[i],ascending=False).head(5).index.tolist())

        labels.append(transposed.sort_values(by=[i],ascending=False).head(5).index.tolist())
    


    return labels



def readCluster(df,num):
    """
    Function to print all tweets within a cluster
    """
    
    groupDf = df[df['clusterLabel'] == num]
    print(groupDf['text_x'].values.tolist())
    





        
