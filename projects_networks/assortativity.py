# -*- coding: utf-8 -*-
"""
Created on : 2019/06/04
Last updated on:
@author: Vincent Bazinet
"""

import numpy as np
from scipy.stats import rankdata
from . import diffusion as dif


def local_assort(A, M, weights=None, pr="multiscale", method="weighted",
                 thorndike=False, return_extra=False, constraint=None):
    '''
    Function to compute the local assortativity in an undirected network

    Parameters
    ----------
    A : ndarray (n,n)
        Adjacency matrix of the network
    M : ndarray (n,) OR tuple of ndarray, each of size (n,)
        Node attributes.
    weights : None or ndarray (n,n)
        Weights to be use to compute the assortativity. rows are the individual
        weight vectors for each nodes. If None, then pagerank vector will be
        used as the weights, with a restart probability given by'pr'.
    pr : Float
        If weights is None, value of the alpha parameter used to compute the
        pagerank vectors. Must be a value between 0 and 1.
    method : str
        Method used to compute the local assortativity. "weighted" computes
        assortativity by computing the weighted Pearson correlation
        Coefficient. "Peel" computes assortativity by standardizing the scalar
        values using the mean and SD of the attributes (Peel et al., 2018).
    thorndike : Boolean
        Correction for possible Restriction of range in correlation computation
        (Thorndike equation II).
    return_extra : Boolean
        Returns dictionary providing extra information regarding the weighted
        distribution of the attributes used to compute the correlation.
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
        if np.all(np.sum(weights, axis=1) == 1) is False:
            raise ValueError("weight vector must sum to 1")
        # Make sure that the weights are stored in (n,n) ndarray (if not None)
        if type(weights) is np.ndarray:
            if weights.shape != (n, n):
                raise TypeError("weights vector must have shape (n,n)")
        else:
            raise TypeError("weights must be stored in a ndarray")

    # Make sure that if thorndike is true, then method is not Peel
    if thorndike is True and method == "Peel":
        raise ValueError("Thorndike correction not available for Peel method")

    # Initialize arrays storing the results
    assort = np.empty(n)
    w_all = np.zeros((n, n))

    # Initialize arrays to store extra information, for when return_extra==True
    if return_extra is True:
        extra_info = np.zeros((n), dtype=object)

    # If weights==None, compute weights using pagerank vector,
    # with given 'pr' value
    if weights is None:
        # Compute weighted vector for every node in the graph
        for i in range(n):
            if pr == "multiscale":
                _, w_all[i, :], _ = dif.getPageRankWeights(A, i, 1)
            else:
                pi, _, _ = dif.getPageRankWeights(A, i, pr)
                w_all[i, :] = pi.reshape(-1)

    # else, your weights are inputed by the user of the function
    else:
        w_all = weights

    if method == "Peel":
        # Compute the zscored values of the attributes
        x_mean = (1/(2*m)) * (np.sum(degree*M))
        x_std = np.sqrt((1/(2*m)) * np.sum(degree * ((M - x_mean)**2)))
        normed_x = (M - x_mean) / x_std

    # Compute local assortativity for each node in the graph
    for i in range(n):

        ti = w_all[i, :]

        weighted_A = ti[:, np.newaxis] * (A / degree[:, np.newaxis])

        if constraint is not None:
            weighted_A = weighted_A * constraint

        x_weights = np.sum(weighted_A, axis=1)
        y_weights = np.sum(weighted_A, axis=0)

        if method == "weighted":

            # Compute the weighted zscores
            if hetero is False:
                x_mean = np.sum(x_weights*M)
                y_mean = np.sum(y_weights*M)
                x_std = np.sqrt(np.sum(x_weights * ((M - x_mean)**2)))
                y_std = np.sqrt(np.sum(y_weights * ((M - y_mean)**2)))

                assort[i] = np.sum(weighted_A * (M-x_mean)[:, np.newaxis] * (M-y_mean)[np.newaxis, :], axis=None)/(x_std*y_std)

            else:
                x_mean = np.sum(x_weights*M)
                y_mean = np.sum(y_weights*N)
                x_std = np.sqrt(np.sum(x_weights * ((M - x_mean)**2)))
                y_std = np.sqrt(np.sum(y_weights * ((N - y_mean)**2)))

                assort[i] = np.sum(weighted_A * (M-x_mean)[:, np.newaxis] * (N-y_mean)[np.newaxis, :], axis=None)/(x_std*y_std)

        else:
            assort[i] = np.sum(weighted_A * normed_x[:, np.newaxis] * normed_x[np.newaxis, :], axis=None)

        if return_extra is True:

            extra_info[i] = {}
            extra_info[i]["x_mean"] = np.sum(x_weights*M)
            extra_info[i]["y_mean"] = np.sum(y_weights*M)
            extra_info[i]["x_stds"] = np.sqrt(np.sum(x_weights * ((M - x_mean)**2)))
            extra_info[i]["y_stds"] = np.sqrt(np.sum(y_weights * ((M - y_mean)**2)))

            extra_info[i]["x_skew"] = np.sum(x_weights * (((M - x_mean)/x_std)**3))
            extra_info[i]["y_skew"] = np.sum(y_weights * (((M - y_mean)/y_std)**3))
            extra_info[i]["x_kurt"] = np.sum(x_weights * (((M - x_mean)/x_std)**4))-3
            extra_info[i]["y_kurt"] = np.sum(y_weights * (((M - y_mean)/y_std)**4))-3

    if thorndike is True:

        x_stds = np.zeros((n))
        for k in range(n):
            x_stds[k] = extra_info[k]['x_stds']
        assort = thorndike_correct(A, M, assort, m, x_stds)

    if return_extra is True:
        return assort, w_all, extra_info

    else:
        return assort, w_all


def globalAssort_fast(A, M, m, edges):
    X = (M[:, np.newaxis]*A)[edges]
    mean = np.mean(X, axis=None)
    std = np.std(X, axis=None)
    norms = (M-mean)/std
    rglobal = np.sum(A * norms[:, np.newaxis] * norms[np.newaxis, :])
    rglobal = rglobal/m

    return rglobal


def globalAssort(A, M, method="pearson", debugInfo=False):
    """
    Function to compute the global assortativity of a BINARY network. This
    function is slighly faster than weighted_assort as it assumes that
    all the weights are 1.
    Parameters
    ----------
        A : (n,n) ndarray
            Adjacency Matrix of the Binary Network of n nodes.
        M  : (n,) or (n,p) ndarray or tuple
            Node Properties. Can be an (n,) ndarray, an (n,p) ndarray,
            where n is the number of nodes and p is the number of attributes
            or a tuple of two (n,) ndarrays.

    Returns
    -------
        If M is a tuple or an (n,) array, returns a scalar value, corresponding
        to the global assortativity measure of the attribute. If the shape is
        (n,p), return a (p,p) matrix of the co-assortativity values between
        each pair of attributes
    """

    if type(M) is np.ndarray:
        n = M.shape[0]

        if M.ndim == 1:

            if method == "spearman":
                # Rank the attribute values
                M = rankdata(M)

            X = M[:, np.newaxis]*A
            mean = np.mean(X[A == 1], axis=None)
            std = np.std(X[A == 1], axis=None)
            norms = (M-mean)/std
            rglobal = np.sum(A * norms[:, np.newaxis] * norms[np.newaxis, :])

        elif M.ndim == 2:

            p = M.shape[1]

            if method == "spearman":
                # Rank the attribute values
                for i in range(p):
                    M[:, i] = rankdata(M[:, i])

            X = np.zeros((n, n, p))
            Y = np.zeros((n, n, p))
            zX = np.zeros((n, p))
            zY = np.zeros((n, p))
            for i in range(p):
                X[:, :, i] = (M[:, i][:, np.newaxis])*A
                Y[:, :, i] = A*(M[:, i][:, np.newaxis])
                mX = np.mean(X[:, :, i][A == 1], axis=None)
                sdX = np.std(X[:, :, i][A == 1], axis=None)
                mY = np.mean(Y[:, :, i][A == 1], axis=None)
                sdY = np.std(Y[:, :, i][A == 1], axis=None)
                zX[:, i] = (M[:, i]-mX)/sdX
                zY[:, i] = (M[:, i]-mY)/sdY

            rglobal = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    rglobal[i, j] = np.sum(A * zX[:, i][:, np.newaxis] * zY[:, j][np.newaxis, :])

        else:
            raise TypeError("M must be of shape (n,) or (n,p)")

    elif type(M) is tuple:
        N = M[1]
        M = M[0]

        if method == "spearman":
            # Rank the attribute values
            M = rankdata(M)
            N = rankdata(N)

        X = M[:, np.newaxis]*A
        Y = A*N[np.newaxis, :]
        mX = np.mean(X[A == 1], axis=None)
        sdX = np.std(X[A == 1], axis=None)
        mY = np.mean(Y[A == 1], axis=None)
        sdY = np.std(Y[A == 1], axis=None)
        zX = (M-mX)/sdX
        zY = (N-mY)/sdY

        rglobal = np.sum(A * zX[:, np.newaxis] * zY[np.newaxis, :])

    denominator = np.sum(A)
    rglobal = rglobal/denominator

    if debugInfo is False:
        return rglobal
    else:
        return rglobal, mean, norms, denominator


def weighted_assort(A, M, N=None):
    '''
    Function to compute the weighted Pearson correlation between the attributes
    of the nodes connected by edges in a network (i.e. weighted assortativity).
    This function also works for binary networks.

    Parameters
    ----------
    A : (n,n) ndarray
        Adjacency matrix of our network.
    M : (n,) ndarray
        Vector of nodal attributes.
    N : (n,) ndarray
        Second vector of nodal attributes (optional)

    Returns
    -------
    ga : float
        Weighted assortativity of our network, with respect to the vector
        of attributes
    '''

    N_nodes = len(A)

    # Normalize the adjacency matrix to make weights sum to 1
    A_norm = A / np.sum(A, axis=None)
    k_norm = np.sum(A_norm, axis=0)

    # Compute the (weighted) mean and standard deviation of our attributes
    M_mean = np.sum(k_norm * M)
    M_sd = np.sqrt(np.sum(k_norm * ((M-M_mean)**2)))

    # Compute the zscores of our attributes and of each edge "endpoints"
    zM = (M - M_mean) / M_sd
    zj = np.repeat(zM[np.newaxis, :], N_nodes, axis=0)

    if N is not None:
        # Do the same thing for our second attribute (if we have a second one)
        N_mean = np.sum(k_norm * N)
        N_sd = np.sqrt(np.sum(k_norm * ((N-N_mean)**2)))

        zN = (N - N_mean) / N_sd
        zi = np.repeat(zN[:, np.newaxis], N_nodes, axis=1)
    else:
        # Otherwise, take the transform of the zj matrix
        zi = zj.T

    # Compute the weighted assortativity as a sum of zscores
    ga = (A_norm * zi * zj).sum()

    return ga


def thorndike_correct(A, M, assortT, m, x_stds):
    x_mean = (1/(2*m)) * (np.sum(np.sum(A, axis=0)*M))
    x_std = np.sqrt((1/(2*m)) * np.sum(np.sum(A, axis=0) * ((M - x_mean)**2)))

    thorndike = np.zeros((len(M)))
    for i in range(len(M)):
        thorndike[i] = x_std*assortT[i]/((((x_std**2)*(assortT[i]**2))+(x_stds[i]**2)-((x_stds[i]**2)*(assortT[i]**2)))**(1/2))

    return thorndike
