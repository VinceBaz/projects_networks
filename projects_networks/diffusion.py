# -*- coding: utf-8 -*-
"""

Functions that are useful for investigating the diffusive architecture of
networks.

Created on : 2020/03/13
Last updated on: 2020/04/25
@author: Vincent Bazinet
"""

import numpy as np
import tqdm
from scipy.linalg import fractional_matrix_power
from scipy.linalg import expm


def transition_matrix(A):
    """
    Function to get the transition matrix (A.K.A. prabability or Markov matrix)
    of a network

    Parameters
    ----------
    A : (n,n) ndarray OR dict
        Either the adjaency matrix of the the network (ndarray) or the full
        dictionary storing a network's information (dict, see load_data.py)

    Return
    -------
    T : (n,n) ndarray
        The transition probability matrix of the network of interest
    """

    if isinstance(A, dict):
        A = A["adj"]

    deg = np.sum(A, axis=0)

    D = np.diag(deg)
    D_inv = np.linalg.matrix_power(D, -1)
    T = np.matmul(D_inv, A)

    return T


def laplacian_matrix(A, version='normal'):
    '''
    Function to get the laplacian matrix of a network.
    A : (n, n) ndarray OR dict
        Either the adjaency matrix of the the network (ndarray) or the full
        dictionary storing a network's information (dict, see load_data.py).
    version : str
        Version of the Laplacian matrix that is to be computed. The available
        options are 'normal' : simple laplacian [L = D-A]; 'rw' : random-walk
        laplacian [L = I - inv(D)A] or 'normalized' : normalized laplacian.
    '''

    if isinstance(A, dict):
        A = A["adj"]

    D = np.diag(np.sum(A, axis=0))
    L = D-A

    if version == 'rw':
        L = np.matmul(np.linalg.matrix_power(D, -1), L)

    if version == 'normalized':
        D_minus_half = fractional_matrix_power(D, -0.5)
        L = np.matmul(D_minus_half, L)
        L = np.matmul(L, D_minus_half)

    return L


def diffuse(network, ts, laplacian='normal', verbose=False):

    if isinstance(network, dict):
        A = network['adj']
    else:
        A = network

    k = len(ts)
    n = len(A)
    pr = np.zeros((k, n, n))

    # Compute the random-walk Laplacian of graph
    L = laplacian_matrix(A, version=laplacian)

    for i in tqdm.trange(k) if verbose else range(k):
        pr[i, :, :] = expm(-1 * ts[i] * L)

    return pr


def random_walk(A, p0, n):
    """
    Function to simulate a simple unbiased random walk on a network.

    Parameters
    ----------
    A : (n,n) ndarray OR dict
        Either the adjaency matrix of the the network (ndarray) or the full
        dictionary storing a network's information (dict).
    p0 : (n,) ndarray OR int
        The initial distribution of random walkers on the network (ndarray). If
        the walk is initiated from a single node of interest, one can use the
        index of this node (int).
    n : int
        Number of iterations to be performed (i.e. the length of the walks)
    """

    if isinstance(A, dict):
        A = A["adj"]

    n = len(A)

    # Get Markov matrix of the network (transposed, in this case)
    W = transition_matrix(A).T

    # Initialize initial probabilities
    if isinstance(p0, np.ndarray):
        F = p0
    else:
        F = np.zeros(n)
        F[p0] = 1

    # Initialize other parameters...
    diff = 1
    it = 1
    Fold = F.copy()

    # Start Power Iteration...
    while it < n:
        F = W.dot(F)
        diff = np.sum((F-Fold)**2)
        if diff < 1e-9:
            print("stationary distribution reached (", it, ")")
            diff = 0
            return F
        Fold = F.copy()
        it += 1

    return F


def getPageRankWeights(A, i, pr, maxIter=1000):
    """
    Function that gives you the personalized PageRank of a node in a network

    Parameters
    ----------
    A : ndarray (n,n)
        Adjacency matrix representation of the network of interest, where
        n is the number of nodes in the network.
    i : int
        Index of the seed node.
    pr : float
        Damping factor (One minus the probability of restart). Must be between
        0 and 1.
    maxIter : int
        Maximum number of iterations to perform in case the algorithm
        does not converge (if reached, then prints a warning).

    Returns
    -------
    F : ndarray (n,)
        The personalized PageRank vector of node i, for the selected damping
        factor
    T : ndarray (n,)
        The integral scores over the PageRank vector scores obtained for
        values of the damping factor. Also called Multiscale PageRank.
    it : int
        Number of iterations
    """

    degree = np.sum(A, axis=1)  # out-degrees
    n = len(A)                  # Number of nodes in the network

    # Check if dimension of inputs are valid
    if degree.ndim != 1 or A.ndim != 2:
        raise TypeError("Dimensions of A and degree must be 2 and 1")

    # Compute the Transition matrix (transposed) of the network
    W = A/degree[:, np.newaxis]
    W = W.T

    # Initialize parameters...
    diff = 1
    it = 1
    F = np.zeros(n)  # PageRank (current)
    F[i] = 1
    Fold = F.copy()  # PageRank (old)
    T = F.copy()     # Pagerank [multiscale]
    oneminuspr = 1-pr

    # Start Power Iteration...
    while diff > 1e-9:

        # Compute next PageRank
        F = pr*W.dot(F)
        F[i] += oneminuspr

        # Compute multiscale PageRank
        T += (F-Fold)/((it+1)*(pr**it))

        it += 1

        diff = np.sum((F-Fold)**2)
        Fold = F.copy()

        if it > maxIter:
            print(i, "max iterations exceeded")
            diff = 0

    return F, T, it


def getPersoPR(A, prs, verbose=False):
    """
    Function to get the Personalized PageRank vectors for all the nodes in a
    given network and for multiple values of probability of restart.

    Parameters
    ----------
    SC : (n,n) ndarray OR dict
        Either the adjacency matrix (numpy array) if a network or the entire
        network itself (python dictionary).
    prs : (k) ndarray OR string
        Either a range of alpha values to use (numpy array) or string
        "multiscale".
    """

    if type(A) is dict:
        A = A["adj"]

    n = len(A)
    k = len(prs)

    if isinstance(prs, np.ndarray):
        perso = np.zeros((k, n, n))
        for c in tqdm.trange(k) if verbose else range(k):
            for i in range(n):
                perso[c, i, :] = getPageRankWeights(A, i, prs[c])[0]

    elif prs == "multiscale":
        perso = np.zeros((n, n))
        for i in range(n):
            _, perso[i, :], _ = getPageRankWeights(A, i, 1)

    return perso
