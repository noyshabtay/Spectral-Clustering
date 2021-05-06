import argparse
from typing import List
import numpy as np
import pandas as pd
import mykmeanssp

MAX_ITER = 300

def squared_euclidian_dist(observations, last_centroid):
    '''
    Arguments: 
    observations- a matrix (list of lists) with data points as rows
    last_centroid- an array of values representing a centroid of a cluster
    Returns:
    the squeared euclidian distance between each row of the matrix and the centroid
    '''
    return np.power(observations-last_centroid, 2).sum(axis=1)

def k_means_pp(k, n, d, observations):
    '''
    Arguments: 
    k- the number of clusters for assignment
    n- number of data points
    d- the number of dimension
    observations- a matrix (list of lists) with data points as rows
    Returns:
    a list of centroid indices for initializing kmeans algorithm according to kmeans++ heuristic
    '''
    np.random.seed(0)
    first_index = np.random.choice(np.arange(n))
    observations = pd.DataFrame(observations)
    indices = [first_index]
    if k == 1:
        return indices
    current_distances = squared_euclidian_dist(observations, observations.iloc[indices[0]])
    for j in np.arange(k-1):
        if j > 0:
            next_distances = squared_euclidian_dist(observations, observations.iloc[indices[j]])
            current_distances = np.minimum(current_distances, next_distances)
        probabilities = np.divide(current_distances, np.sum(current_distances))
        next_index = np.random.choice(np.arange(n), p=probabilities)
        indices.append(next_index)
    return indices

def kmeans_api(observations, k, n, d):
    '''
    Arguments: 
    X- a matrix with data points as rows
    k- the number of clusters for assignment
    n- number of data points
    d- the number of dimension
    Effects:
    calls kmeans_runner implemented in C
    Returns:
    a list of data points and their corresponding cluster indices
    a list of clusters where each cluster is an array of data point indices
    '''
    # k_means++ init
    indices = k_means_pp(k, n, d, observations)

    #DataFrames to 2-dimensional lists for C-API
    observations_list = observations.tolist()

    # k_means clustring
    try:
        answer = mykmeanssp.kmeans_runner(k, n, d, MAX_ITER, indices, observations_list)
    except:
        raise Exception("There was an OS allocation problem or a C-API failure while running the kmeans module! All used memory freed successfully.")
    return answer[0], np.array(answer[1])
