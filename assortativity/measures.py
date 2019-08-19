# -*- coding: utf-8 -*-
"""
Created on:        Tue Jun 4 19:44:05 2019
Last updated on:
@author: Vincent Bazinet
"""

import numpy as np
import scipy.sparse as sparse


def localAssort(A, M, weights=None, pr="multiscale", method="weighted",
                thorndike=True, return_extra=False, constraint=None):
    '''
    Function to compute the local assortativity in an undirected network
    INPUTS:
    A           ->  Adjacency matrix - Numpy Array, size:(n,n)
    M           ->  Node Properties - Numpy Array, size:(n) OR Tuple of 2 Numpy Arrays, size:(n)
    weights     ->  Weights to be use to compute the assortativity. If None,
                    then pagerank vector will be used as the weights, with
                    a restart probability given by 'pr' - None or Numpy Array, size:(n,n)
                    where row n corresponds to the weight vector of node n
    pr          ->  If weights is None, value of the alpha parameter used to compute
                    the pagerank vectors - Float, between 0 and 1
    method      ->  Method used to compute the local assortativity. "weighted" computes
                    assortativity by computing the weighted Pearson correlation Coefficient.
                    "Peel" computes assortativity by standardizing the scalar values using
                    the mean and SD of the attributes (Peel et al., 2018)
    thorndike   ->  Correction for possible Restriction of range in correlation computation
                    (Thorndike equation II)
    '''

    # Check if its heterogenous or not
    if type(M) is tuple:
        N = M[1]
        M = M[0]
        hetero = True
    elif type(M) is np.ndarray:
        hetero = False
    else:
        raise TypeError("Node Properties must be stored in a ndarray")

    n = len(M)                  # Nb of nodes
    m = np.sum(A)/2             # Nb of edges (divided by two when undirected)
    degree = np.sum(A, axis=0)  # Degree of nodes (or Strength for weighted)

    if weights is not None:
        # Make sure that the weights sum to 1
        if np.all(np.sum(weights, axis=1)==1) is False:
            raise ValueError("weight vector must sum to 1")
        # Make sure that the weights are stored in a (n,n) ndarray (if not None)
        if type(weights) is np.ndarray:
            if weights.shape is not (n,n):
                raise TypeError("weights vector must have shape (n,n)")
        else:
            raise TypeError("weights must be stored in a ndarray")

    #Initialize arrays storing the results
    assort = np.empty(n)
    w_all = np.zeros((n,n))

    #Initialize arrays to store extra information, for when return_extra==True
    x_means = np.zeros((n))
    y_means = np.zeros((n))
    x_stds = np.zeros((n))
    y_stds = np.zeros((n))

    #If weights are None, you compute the weights using the pagerank vector, with given 'pr' value
    if weights is None:
        #Compute weighted vector for every node in the graph
        for i in range(n):
            if pr=="multiscale":
                _,w_all[i,:],_ = calculateRWRrange(sparse.csc_matrix(A), degree, i, np.array([1]), n)
            else:
                pi,_,_ = calculateRWRrange(sparse.csc_matrix(A), degree, i, np.array([pr]), n)
                w_all[i,:] = pi.reshape(-1)

    #else, your weights are inputed by the user of the functio0n
    else:
        w_all = weights

    if method is "Peel":
        #Compute the zscored values of the attributes
        x_mean = (1/(2*m)) * (np.sum(degree*M))
        x_std =  np.sqrt((1/(2*m)) * np.sum(degree * ((M - x_mean)**2)))
        normed_x = (M - x_mean)/ x_std

    #Compute local assortativity for each node in the graph
    for i in range(n):

        ti = w_all[i,:]

        weighted_A = ti[:,np.newaxis] * (A / degree[:,np.newaxis])

        if constraint is not None:
            weighted_A = weighted_A * constraint

        if method is "weighted":

            #Compute the weighted zscores
            if hetero is False:
                x_mean = np.sum((np.sum(weighted_A, axis=1)*M))
                y_mean = np.sum((np.sum(weighted_A, axis=0)*M))
                x_std = np.sqrt(np.sum(np.sum(weighted_A, axis=1) * ((M - x_mean)**2)))
                y_std = np.sqrt(np.sum(np.sum(weighted_A, axis=0) * ((M - y_mean)**2)))

            else:
                x_mean = np.sum((np.sum(weighted_A, axis=1)*M))
                y_mean = np.sum((np.sum(weighted_A, axis=0)*N))
                x_std = np.sqrt(np.sum(np.sum(weighted_A, axis=1) * ((M - x_mean)**2)))
                y_std = np.sqrt(np.sum(np.sum(weighted_A, axis=0) * ((N - y_mean)**2)))

            x_means[i] = x_mean
            y_means[i] = y_mean
            x_stds[i] = x_std
            y_stds[i] = y_std

            assort[i] = np.sum(weighted_A * (M-x_mean)[:,np.newaxis] * (M-y_mean)[np.newaxis,:], axis=None)/(x_std*y_std)

        else:
            assort[i] = np.sum(weighted_A * normed_x[:,np.newaxis] * normed_x[np.newaxis,:], axis=None)

    if thorndike==True:
        assort = thorndike_correct(A, M, assort, m, x_stds)

    if return_extra==True:
        return assort, w_all, x_means, y_means, x_stds, y_stds

    else:
        return assort, w_all


def globalAssort(A, M, debugInfo=False):

    m = np.sum(A)/2
    degree = np.sum(A, axis=1).astype(int)
    edge_M = np.zeros((2*int(m)))
    count=0
    for i in range(len(M)):
        for j in range(degree[i]):
            edge_M[count] = M[i]
            count+=1

    x_mean = (1/(2*m)) * (np.sum((np.asarray(np.sum(A, axis=1)).reshape(-1)*M)))
    M_norm = (M-x_mean)/np.std(edge_M)

    np.sum(A, axis=1)

    rglobal = np.sum(A * M_norm[:,np.newaxis] * M_norm[np.newaxis,:])

    denominator = 2*m

    rglobal = rglobal/denominator

    if debugInfo==False:
        return rglobal
    else:
        return rglobal, x_mean, M_norm, denominator

def weightedAssort(A, M):

    Size = len(A)
    return(corr(np.tile(M, (Size,1)).reshape(-1), np.tile(M, (Size,1    )).T.reshape(-1), A.reshape(-1)))

# calculate the stationary distributions of a random walk with restart
# for different probabilties of restart (using the pagerank as a function
# approach)
def calculateRWRrange(A, degree, i, prs, n, trans=True, maxIter=1000):

    #Use maximum 'pr' parameter for this computation
    pr = prs[-1]

    #Divide each row 'i' by the degree of node 'i'
    D = sparse.diags(1./degree, 0, format='csc')
    W = D.dot(A)

    #Initialize parameters
    diff = 1
    it = 1
    F = np.zeros(n)
    Fall = np.zeros((n, len(prs)))
    F[i] = 1
    Fall[i, :] = 1
    Fold = F.copy()
    T = F.copy()

    #Get the transpose
    if trans:
        W = W.T   #W is the Markov Matrix (a.k.a. Transition Probability Matrix)

    oneminuspr = 1-pr


    while diff > 1e-9:
        F = pr*W.dot(F)
        F[i] += oneminuspr
        Fall += np.outer((F-Fold), (prs/pr)**it)
        T += (F-Fold)/((it+1)*(pr**it))

        diff = np.sum((F-Fold)**2)
        it += 1
        if it > maxIter:
            print(i, "max iterations exceeded")
            diff = 0
        Fold = F.copy()

    return Fall, T, it


def random_walk_weighted(A, pr, seed, maxIter=1000):

    weights = np.zeros((len(A)))

    strength = np.sum(A, axis=1)

    n=seed
    for t in range(maxIter):
        probs = A[n,:]/strength[n]

def thorndike_correct(A, M, assortT, m, x_stds):
    x_mean = (1/(2*m)) * (np.sum(np.sum(A, axis=0)*M))
    x_std =  np.sqrt((1/(2*m)) * np.sum(np.sum(A, axis=0) * ((M - x_mean)**2)))

    thorndike=np.zeros((len(M)))
    for i in range(len(M)):
        thorndike[i] = x_std*assortT[i]/((((x_std**2)*(assortT[i]**2))+(x_stds[i]**2)-((x_stds[i]**2)*(assortT[i]**2)))**(1/2))

    return thorndike

def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))
