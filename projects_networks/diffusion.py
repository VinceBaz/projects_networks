# -*- coding: utf-8 -*-
"""

Functions that are useful for investigating the diffusive architecture of
networks.

Created on : 2020/03/13
Last updated on: 2021/02/21
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

    degree = np.sum(A, axis=1)
    T = A/degree[:, np.newaxis]

    return T


def laplacian_matrix(A, version='normal'):
    '''
    Function to get the laplacian matrix of a network.

    Parameters
    ----------
    A : (n, n) ndarray OR dict
        Either the adjaency matrix of the the network (ndarray) or the full
        dictionary storing a network's information (dict, see load_data.py).
    version : str
        Version of the Laplacian matrix that is to be computed. The available
        options are 'normal' : simple laplacian [L = D-A]; 'rw' : random-walk
        laplacian [L = I - inv(D)A] or 'normalized' : normalized laplacian.

    Returns
    -------
    L : (n, n) ndarray
        The Laplacian matrix of the graph
    '''

    if version not in ['normal', 'rw', 'normalized']:
        raise ValueError(("The versions of the laplacian matrix are either"
                          "\'normal\', \'rw\' or \'normalized\'"))

    if isinstance(A, dict):
        A = A["adj"]

    # convert adjacency matrix to float
    A = A.astype(float)

    D = np.diag(np.sum(A, axis=1))
    L = D-A

    if version == 'rw':
        L = np.matmul(np.linalg.matrix_power(D, -1), L)

    if version == 'normalized':
        D_minus_half = fractional_matrix_power(D, -0.5)
        L = np.matmul(D_minus_half, L)
        L = np.matmul(L, D_minus_half)

    return L


def diffuse(network, ts, laplacian='normal', linear_increments=False,
            verbose=False):
    '''
    Function to simulate diffusion processes on a graph.

    Parameters
    ----------
    network: (n, n) ndarray OR dict
        Either the adjaency matrix of the the network (ndarray) or the full
        dictionary storing a network's information (see load_data.py).
    laplacian : 'normal', 'normalized' or 'rw'
        The type of Laplacian matrix to use for the diffusion process. If 'rw'
        is choosen, the diffusion process will be relying on the random-walk
        laplacian to generate what has been refered to has the 'heat-kernel
        PageRank' of the network [A1].
    linear_increments : boolean
        If True, the time increments are assumed to be linear, and the
        diffusion probabilities are computed more rapidly by making use of the
        following equality: e^(tA)e^(sA) = e^((t+s)A).

    References
    ----------
    .. [A1] Chung, F. (2007). The heat kernel as the pagerank of a graph.
    Proceedings of the National Academy of Sciences, 104(50), 19735-19740.
    '''

    if isinstance(network, dict):
        A = network['adj']
    else:
        A = network

    k = len(ts)
    n = len(A)
    pr = np.zeros((k, n, n))

    # Compute the random-walk Laplacian of the graph
    L = laplacian_matrix(A, version=laplacian)

    if linear_increments:
        # Check if increments are linear
        for i in range(2, len(ts)):
            diff1 = ts[i] - ts[i-1]
            diff2 = ts[i-1] - ts[i-2]
            if not np.isclose(diff1, diff2):
                raise ValueError("time increments must be linear")

        delta = (ts.max() - ts.min()) / (len(ts)-1)
        exp1A = expm(-L)
        pr[0, :, :] = exp1A
        expdt = expm(-delta * L)

        for i in tqdm.trange(1, k) if verbose else range(1, k):
            pr[i, :, :] = np.matmul(pr[i-1, :, :], expdt)

    else:
        for i in tqdm.trange(k) if verbose else range(k):
            pr[i, :, :] = expm(-ts[i] * L)

    return pr


def random_walk(A, p0, n):
    """
    Function to simulate a simple unbiased random walk on a network.

    Parameters
    ----------
    A : (n,n) ndarray OR dict
        Either the adjaency matrix of the network (ndarray) or the full
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


def getPageRankWeights(A, i, pr, T=None, degree=None, multiscale=False,
                       maxIter=1000):
    """
    Function that gives you the personalized PageRank of a node in a network.

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
    T : ndarray (n,n)
        Transition probability matrix of the adjacency matrix. Adding this
        matrix will speed up the process if one needs to run this function
        thousands of time.
    degree : ndarray (n,)
        Vector storing the degree of the nodes in the network. Adding this
        vector as a parameter will speed up the process if one needs to run
        this function thousands of time.
    maxIter : int
        Maximum number of iterations to perform in case the algorithm
        does not converge (if reached, then prints a warning).

    Returns
    -------
    F : ndarray (n,)
        The personalized PageRank vector of node i, for the selected damping
        factor
    M : ndarray (n,)
        The integral scores over the PageRank vector scores obtained for
        values of the damping factor. Also called Multiscale PageRank.
    it : int
        Number of iterations
    """

    # Number of nodes in the network
    n = len(A)

    # out-degrees (or stength if weighted)
    if degree is None:
        degree = np.sum(A, axis=1)

    # Compute the Transition matrix (transposed) of the network
    if T is None:
        T = A/degree[:, np.newaxis]
        T = T.T

    # Check if dimension of inputs are valid
    if degree.ndim != 1 or A.ndim != 2:
        raise TypeError("Dimensions of A and degree must be 2 and 1")

    # Initialize parameters
    diff = 1
    it = 1
    F = np.zeros(n)  # PageRank (current)
    F[i] = 1
    Fold = F.copy()  # PageRank (old)
    oneminuspr = 1-pr

    # Pagerank [multiscale]
    if multiscale:
        M = F.copy()

    # Start Power Iteration...
    while diff > 1e-9:

        # Compute next PageRank
        F = pr*T.dot(F)
        F[i] += oneminuspr

        # Compute multiscale PageRank
        if multiscale:
            M += (F-Fold)/((it+1)*(pr**it))

        it += 1

        diff = np.sum((F-Fold)**2)
        Fold = F.copy()

        if it > maxIter:
            print(i, "max iterations exceeded")
            diff = 0

    if multiscale:
        return M, it
    else:
        return F, it


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

    # Compute transition probability matrix
    T = transition_matrix(A).T
    degree = np.sum(A, axis=1)

    if isinstance(prs, np.ndarray):
        perso = np.zeros((k, n, n))
        for c in tqdm.trange(k) if verbose else range(k):
            for i in range(n):
                perso[c, i, :], _ = getPageRankWeights(A, i, prs[c], T=T,
                                                       degree=degree)

    elif prs == "multiscale":
        perso = np.zeros((n, n))
        for i in range(n):
            perso[i, :], _ = getPageRankWeights(A, i, 1, T=T, degree=degree,
                                                multiscale=True)

    return perso
