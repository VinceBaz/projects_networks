# -*- coding: utf-8 -*-
"""
Created on : 2020/03/13
Last updated on: 2020/03/13
@author: Vincent Bazinet
"""

import numpy as np


def getPageRankWeights(A, i, pr, maxIter=1000):
    '''
    Function that gives you the personalized pagerank of node i in network A
    INPUTS:
    A       -> Adjacency matrix representation of your network. Dtype: (n, n)
               ndarray where n is the number of nodes in the network
    i       -> Index of node of interest
    pr      -> Probability of restart
    OUTPUTS:
    F 		-> The Pagerank weights for each node
    T       -> The integral scores of a distribution of pagerank weights
               [Multiscale PageRank]
    it      -> Number of iteration
    '''

    degree = np.sum(A, axis=1)  # out-degrees
    n = len(A)                  # Number of nodes in the network

    # Check if dimension of inputs are valid
    if degree.ndim != 1 or A.ndim != 2:
        raise TypeError("Dimensions of A and degree must be 2 and 1")

    # Divide each row 'i' by the degree of node 'i', then get the transpose
    W = A/degree[:, np.newaxis]
    W = W.T  # Gives you the Markov Matrix of the network

    # Initialize parameters...
    # F      -> The PageRank weights (current)
    # Fold   -> The PageRank weights (old)
    diff = 1
    it = 1
    F = np.zeros(n)
    F[i] = 1
    Fold = F.copy()
    T = F.copy()
    oneminuspr = 1-pr

    # Start Power Iteration...
    while diff > 1e-9:
        F = pr*W.dot(F)
        F[i] += oneminuspr
        T += (F-Fold)/((it+1)*(pr**it))
        diff = np.sum((F-Fold)**2)
        it += 1
        if it > maxIter:
            print(i, "max iterations exceeded")
            diff = 0
        Fold = F.copy()

    return F, T, it


def getPersoPR(A, prs):
    '''
    Function to get the Personalized PageRank vectors for all the nodes in a
    given network and for multiple values of probability of restart.
    -----
    INPUTS
    SC [(n,n) ndarray] OR dict  : either the adjacency matrix (numpy array)
                                  if a network or the entire network itself
                                  (python dictionary).
    prs [(k) ndarray] or string : either a range of alpha values to use
                                  (numpy array) or string "multiscale".
    '''

    if type(A) is dict:
        A = A["adj"]

    n = len(A)

    if isinstance(prs, np.ndarray):
        perso = np.zeros((len(prs), n, n))
        for pr, c in zip(prs, range(len(prs))):
            for i in range(n):
                perso[c, i, :] = getPageRankWeights(A, i, pr, maxIter=1000)[0]

    elif prs == "multiscale":
        perso = np.zeros((n, n))
        for i in range(n):
            _, perso[i, :], _ = getPageRankWeights(A, i, 1, maxIter=1000)

    return perso
