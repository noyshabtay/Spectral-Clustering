import argparse
import itertools
import math
import numpy as np
import random as rd
from sklearn import preprocessing, datasets
from datetime import datetime   
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os

epsilon = 0.0001

def method_handler(fn):
    '''
    Dev. Helper Function
    Wraps a given method (by putting @method_handler above it's definition) and prints the time it takes to run it
    '''
    def wrapper(*args, **kwargs):
        print("Running ", fn.__name__)
        start_time = datetime.now()
        response = fn(*args, **kwargs)
        end_time = datetime.now()
        print (f"***Finished {fn.__name__} in {(end_time-start_time).total_seconds()} seconds", end='\n\n')
        return response
    return wrapper

def spectral_clustering_algorithm(X, k = None) :
    '''
    Arguments: 
    X- data points in a numpy array
    k- number of clusters to assign
    Returns:
    An array of assigned data points to Spectral clusters
    '''
    n = len(X)
    W = weighted_adjacency_matrix(n, X)
    D = diagonal_degree_matrix(W)
    L_norm = normalized_graph_laplacian(n, D, W)
    k, U = k_eigenvectors(n, L_norm, k)
    T = create_T(U)
    clusters, assingment = kmeans(T, k)
    return k, clusters, assingment

def weighted_adjacency_matrix(n, A):
    '''
    Arguments: 
    n- number of points
    A- data points in a numpy array
    Returns:
    Affinity matrix
    (square matrix where for every i W[i][i]==0 & for every j!=i W[i][j]==exp(-0.5*l2-norm(Ai-Aj)))
    '''
    results = np.zeros((n, n))
    for i in range(n):
        results[i] = np.exp(-0.5 * np.linalg.norm(X[i] - X, axis=1))
    np.fill_diagonal(results ,0)
    return results

def diagonal_degree_matrix(W):
    '''
    Arguments: 
    W- affinity matrix
    Returns:
    Diagonal degree matrix
    (where the i-th element along the diagonal equals to the sum of the i-th row of W)
    in the power of -0.5
    '''
    return np.diag(np.power(np.sum(W, axis=1),-0.5))
        
def normalized_graph_laplacian(n, D, W):
    '''
    Arguments: 
    W- affinity matrix
    D- diagonal degree matrix in the power of -0.5
    n- the size of the square matrices W & D
    Returns:
    A normalized Laplacian matrix
    '''
    return np.identity(n) - np.linalg.multi_dot([D,W,D])

def k_eigenvectors(n, L_norm, k): 
    '''
    Arguments: 
    n- number of data points
    L_norm- a normalized Laplacian matrix
    k- Optional: number of clusters
    Returns:
    the determined k and the first k eigenvectors of L_norm 
    '''
    A_roof, Q_roof = qr_iteration_algorithm(n, L_norm)
    eigenvalues = A_roof.diagonal()
    sorted_eigenvalues, idx = sort_eigen(eigenvalues) 
    if not k:
        k = determine_k(n, sorted_eigenvalues)
    vectors = get_k_eigenvectors(Q_roof, k, idx)
    return k, vectors

def sort_eigen(eigenvalues):
    '''
    Arguments: 
    eigenvalues - an array of eigenvalues
    Returns:
    A sorted array of eigenvalues and the sorted original indices
    '''
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    return eigenvalues, idx

def determine_k(n, eigenvalues):
    '''
    Arguments: 
    n- number of data points
    eigenvalues- a sorted array of eigenvalues
    Returns:
    The number of clusters chosen by using the eigengap heuristic
    '''
    differences = np.diff(eigenvalues)[:int(n/2)]
    return np.argmax(differences) + 1 
    
def get_k_eigenvectors(Q_roof, k, idx):
    '''
    Arguments: 
    Q_roof- a matrix composed of eigenvectors
    k- the number of requested vectors
    idx- an array of indices of the sorted eigenvalues
    Returns:
    An array of k eigenvectors sorted by their corresponding eigenvalues
    '''
    return Q_roof[:,idx[:k]]

def mgs_algorithm(n, A):
    '''
    Arguments: 
    n- number of data points
    A- Matrix
    Returns:
    A decomposition of A into two matrixes - orthogonal Q & upper triangular R, such that A = QR
    '''
    U = A.copy()
    R = np.zeros((n, n))
    Q = np.zeros((n, n))
    for i in np.arange(n):
        R[i,i] = np.linalg.norm(U[:,i])
        Q[:,i] = np.divide(U[:,i], R[i,i])
        R[i,i+1:] = (Q[:,i]) @ (U[:,i+1:])
        U[:,i+1:] = U[:,i+1:] - (Q[:,i,np.newaxis] @ R[np.newaxis,i,i+1:])
    #in case of division by zero or other numerical problem. Shouldn't happen with this implementation and data.
    Q[np.isnan(Q)] = 0.0
    Q[np.isinf(Q)] = 0.0
    return Q, R

def qr_iteration_algorithm(n, L_norm):
    '''
    Arguments: 
    L_norm - a normalized laplacian matrix
    n- the shape of L_norm
    Returns:
    A_roof- a matrix with approximated L_norm's eigenvalues on its diagonal 
    Q_roof- a matrix composed of approximated L_norm's eigenvectors as columns
    '''
    A_roof = L_norm.copy()
    Q_roof = np.identity(n)
    for i in np.arange(n):
        Q,R = mgs_algorithm(n, A_roof)
        A_roof = R.dot(Q) 
        if (np.abs(np.abs(Q_roof) - np.abs(Q_roof.dot(Q))) < epsilon).all():
            return A_roof, Q_roof #convergence answer
        Q_roof = Q_roof.dot(Q)
    return A_roof, Q_roof #non-convergence answer

def create_T(U):
    '''
    Arguments: 
    U- a matrix composed of eigenvectors
    Returns:
    A matrix where each row normalized by its norm
    '''
    T = preprocessing.normalize(U, norm='l2', axis=1)
    return T

def kmeans(T, k):
    '''
    Arguments: 
    T- a matrix whose rows are to be clustered
    k- number of requested clusters
    Returns:
    data- a list of data points and their assigned cluster index
    assignment- an array of clusters where each cluster is an array of its assigned indices
    '''
    from kmeans_pp import kmeans_api
    
    data, assignment = kmeans_api(observations = T, k=k, n=T.shape[0], d= T.shape[1])
    return data, assignment

def create_plot(pdf, X, clusters, d, title):
    '''
    Arguments: 
    pdf- a PdfPages context manager
    X- a matrix with data points as rows
    clusters- the assignment of each data point to a cluster index
    d- the number of dimension (number of columns in X)
    title- the plot's requested title
    Effects:
    saves a PDF file with the requested plots
    for a better view and analysis, each plot is displayed in large size and high quality on its own page.
    Returns:
    None
    '''
    fig = plt.figure(figsize=(8,7))
    if d==2: 
        df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], clusters = clusters))
        ax = fig.add_subplot()
        ax.scatter(df['x'], df['y'], c=df['clusters']) 
    else:
        df = pd.DataFrame(dict(x=X[:,0], y=X[:,1], z=X[:,2], clusters = clusters))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(df['x'], df['y'], df['z'], c=df['clusters']) 
        ax.set_zlabel('Z')
    ax.set_xlabel('\nX')
    ax.set_ylabel('Y')
    plt.title(title)
    pdf.savefig()

def visualize_clusters(X, k, k_blobs, d, spectral_assignment, kmeans_assignment, Y):
    '''
    Arguments: 
    X- a matrix with data points as rows
    k- the number of clusters for assignment
    k_blobs- the number of clustered used the generate the randomized data
    d- the number of dimensions (number of columns in X)
    spectral_assignment- assignment to clusters by the spectral algorithm
    kmeans_assignment - assignment to clusters by the kmeans algorithm
    Y- assignment by the generated data
    Effects:
    calculates Jaccard measure for both algorithms and calls a plot creation method
    Returns:
    None
    '''
    n = len(X)
    with PdfPages(os.path.join(os.getcwd(),'clusters.pdf')) as pdf:
        firstPage = plt.figure()
        firstPage.clf()
        spectral_jaccard = jaccard_measure(spectral_assignment, Y, n)
        kmeans_jaccard = jaccard_measure(kmeans_assignment, Y, n)
        txt = f'Data was generated from the values:\nn = {n} , k = {k_blobs}\nThe k that was used for both algorithms was: {k}\nThe Jaccard measure for Spectral Clustering: {"%.2f" % spectral_jaccard}\nThe Jaccard measure for K-means: {"%.2f" % kmeans_jaccard}'
        firstPage.text(0.5,0.5,txt, transform=firstPage.transFigure, size=11, ha="center")
        pdf.savefig()
        create_plot(pdf, X, spectral_assignment, d, 'Normalized Spectral Clustering')
        create_plot(pdf, X, kmeans_assignment, d, 'K-means')
    plt.close()

def jaccard_measure(assignment, Y, n):
    '''
    Arguments: 
    assignment- an array of clusters with the data points indices in them
    Y- an assignment created by the original data generation
    n- number of assigned data points
    Returns:
    the jaccard measure of the assignment
    '''
    mutual = unmutual = 0
    for i, j in itertools.combinations(range(n), 2):
        is_same_assignment = assignment[i] == assignment[j]
        is_same_Y = Y[i] == Y[j]
        if is_same_assignment and is_same_Y:
            mutual += 1
        elif (is_same_assignment and (not is_same_Y)) or ((not is_same_assignment) and is_same_Y):
            unmutual += 1
    return float(mutual) / (mutual + unmutual)

def textual_output(X, Y, n, k, spectral_clusters, kmeans_clusters):
    '''
    Arguments: 
    X- a matrix with data points as rows
    Y- an assignment created by the original data generation
    n- number of assigned data points
    spectral_clusters- the clustered data points by spectral clustering
    kmeans_clusters- the clustered data points by kmeans
    Effects:
    prints the number of clusters, and each cluster data points indices assignement to clusters.txt
    prints the generated data points in order with their true label.
    Returns:
    None
    '''
    with open('Data.txt','w') as file:               
        for i in range(n):
            line = ",".join(map(str, X[i])) + f",{Y[i]}"
            print(line, file=file)
    with open('Clusters.txt', 'w') as file:
        print(k, file=file)
        for i in range(k):
            line = ",".join(map(str, spectral_clusters[i]))
            print(line, file=file)
        for i in range(k):
            line = ",".join(map(str, kmeans_clusters[i]))
            print(line, file=file)

def parse_arguments():
    '''
    Arguments: 
    None
    Effects:
    parses command line arguments
    Returns:
    namespace of arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("K", type=int, help="Number of clusters")
    parser.add_argument("N", type=int, help="Number of observations")
    parser.add_argument('--Random', dest='random', action='store_true')
    parser.add_argument('--no-Random', dest='random', action='store_false')
    parser.set_defaults(random=True) 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    #Max capacity determined by an average of 5 runs on the Nova server
    #Running time can vary greatly (for better or worse) due to loads on the server
    class dim_2:
        k_max = 20
        n_max = 420
    
    class dim_3:
        k_max = 20
        n_max = 420

    print(f"The max capacity of this program for 2D data:\nk: {dim_2.k_max} n:{dim_2.n_max}")
    print(f"The max capacity of this program for 3D data:\nk: {dim_3.k_max} n:{dim_3.n_max}")


    #Arguments
    args = parse_arguments()
    k, n, random = args.K, args.N, args.random

    d = rd.randint(2,3)
    if d == 2:
        max_capacity = dim_2
    else:
        max_capacity = dim_3

    #Validate arguments
    if random == False and (not isinstance(n,int)) or (not isinstance(k,int)):
        raise Exception("Not all command line arguments have the right type")
    if random == False and ((n < 1) or (k < 1) or (k >= n)):
        raise Exception("The values set for n and k are invalid")
    
    #Generate data points
    if random:
        k = None 
        k_blobs = rd.randint(int(max_capacity.k_max/2), max_capacity.k_max)
        n = rd.randint(int(max_capacity.n_max/2), max_capacity.n_max)
    else:
        k_blobs = k
    X, Y = datasets.make_blobs(n_samples = n, n_features = d, centers = k_blobs)

    #Spectral Clustering Algorithm
    k, spectral_clusters, spectral_assignment = spectral_clustering_algorithm(X, k)

    #Kmeans
    kmeans_clusters, kmeans_assignment = kmeans(X, k)

    #Plots 
    visualize_clusters(X, k, k_blobs, d, spectral_assignment, kmeans_assignment, Y)

    #Output to file
    textual_output(X.tolist(), Y, n, k, spectral_clusters, kmeans_clusters)