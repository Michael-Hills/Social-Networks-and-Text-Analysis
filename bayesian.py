import pandas as pd
import numpy as np
from cluster import *
from hyperopt import fmin,space_eval,partial,Trials,tpe,STATUS_OK,hp
from ast import literal_eval




def scoreClusters(clusterer, threshold):
    """
    Function to calculate the cluster confidence cost
    """
    print("Scoring...")
    
    cluster_labels = clusterer.labels_
    label_count = len(np.unique(cluster_labels))
    cost = (np.count_nonzero(clusterer.probabilities_ < threshold)/len(cluster_labels))
    
    return label_count, cost

def scoreClustersWithoutNoise(clusterer,threshold):
    """
    Function to calculate cluster confidence cost without points identified as noise
    """

     
    cluster_labels = clusterer.labels_

    #exclude noise points (-1) from the cluster labels
    clustered_points_mask = cluster_labels != -1
    cluster_labels_without_noise = cluster_labels[clustered_points_mask]

    #count the number of non-noise clusters
    label_count = len(np.unique(cluster_labels_without_noise))

    #compute the cost using only non-noise points
    non_noise_probabilities = clusterer.probabilities_[clustered_points_mask]
    cost = np.count_nonzero(non_noise_probabilities < threshold) / len(cluster_labels_without_noise)

    return label_count, cost



def objective(params, embeddings):
    """
    Objective function to minimise
    """

    print("Clustering...")
    
    #create the clusters from the embeddings
    clusters = generateClusters(embeddings, 
                                 n_neighbors = params['n_neighbors'], 
                                 min_dist = params['min_dist'],
                                 n_components = params['n_components'], 
                                 min_cluster_size = params['min_cluster_size'],
                                 min_samples = params['min_samples'])
    

    #calculate the cost
    label_count, cost = scoreClusters(clusters, threshold = 0.20)
    
    
    return {'loss': cost, 'label_count': label_count, 'status': STATUS_OK}



def bayesianSearch(embeddings, space, maxEvals):
    """
    Function to perform Bayesian search by minimising the objective function
    """

    print("Starting search...")
    
    #run trials
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings)
    best = fmin(fmin_objective, 
                space = space, 
                algo=tpe.suggest,
                max_evals=maxEvals, 
                trials=trials)


    #print the best parameters
    best = space_eval(space, best)
    print ('best:')
    print (best)
    print (f"label count: {trials.best_trial['result']['label_count']}")
    

    #generate the clusters of the best
    bestClusters = generateClusters(embeddings, 
                                      n_neighbors = best['n_neighbors'], 
                                      min_dist = best['min_dist'],
                                      n_components = best['n_components'], 
                                      min_cluster_size = best['min_cluster_size'],
                                      min_samples = best['min_samples'])
    
    return best, bestClusters, trials



def getBayesianScores(df,group):
    """
    Function that runs the bayesian search
    """  

    groupDf = df[df['author_id_y'] == group]
    vectors = groupDf['textVector'].values.tolist()  

    #the search space of the search
    hspace = {
        "n_neighbors": hp.choice('n_neighbors',range(2,32,2)),
        "min_dist":hp.choice('min_dist',np.arange(0,0.14,0.02)),
        "n_components": hp.choice('n_components',range(5,11)),
        "min_cluster_size": hp.choice('min_cluster_size',range(10,40,5)),
        "min_samples": hp.choice('min_samples',range(1,6))}

    best_params_use, best_clusters_use, trials_use = bayesianSearch(vectors,space=hspace,maxEvals=1)