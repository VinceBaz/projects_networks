# -*- coding: utf-8 -*-
"""
Created on : 2019/11/20
Last updated on : 2020/05/23
@author: Vincent Bazinet
"""

import numpy as np
import pickle
from scipy.stats import pearsonr
from projects_networks import load_data


def mask(network, other_networks=None, with_diag=False, type="all"):
    '''
    Function to retrieve a relevant mask of our networks

    Parameters
    ----------
    network : dict
        Dictionary of the network for which we need a mask.
    other_networks : list of dicts
        List of dictionaries with additional networks with same size as
        network.
    with_diag : Boolean
        True is we want to keep the diagonal in the mask. False is we want to
        remove it.
    type : str
        Type of mask to be used. The available options are "all", "non_zero"
        for having a mask of all the edges with non_zero values, or "zero".

    Returns
    -------
    mask : ndarray (n,n)
        Mask of the network

    '''

    if isinstance(network, dict):
        A = network["adj"]
    else:
        A = network

    n = len(A)
    mask = np.zeros((n, n), dtype="bool")
    mask[:] = True

    if with_diag is False:
        mask[np.diag_indices(n)] = False

    if type == "all":
        return mask

    elif type == "non_zero":
        mask[A == 0] = False

        if other_networks is not None:
            for i in range(len(other_networks)):
                mask[other_networks[i]["adj"] == 0] = False

    elif type == "zero":
        mask[A != 0] = False

        if other_networks is not None:
            for i in range(len(other_networks)):
                mask[other_networks[i]["adj"] != 0] = False

    return mask


def scale(values, vmin, vmax, axis=None):

    s = (values - values.min(axis=axis)) / (values.max(axis=axis) - values.min(axis=axis))
    s = s * (vmax - vmin)
    s = s + vmin

    return s


def isBipartite(A):
    '''
    Function to check if a network is Bipartite

    Parameters
    ----------
    A : ndarray (n,n)
        Adjacency matrix of the network
    '''

    n = len(A)

    # Create a color array to store colors assigned to all veritces. Vertex
    # number is used as index in this array. The value '-1' of  colorArr[i] is
    # used to indicate that no color is assigned to vertex 'i'. The value 1 is
    # used to indicate first color is assigned and value 0 indicates second
    # color is assigned.
    colorArr = np.zeros((n))-1

    # Assign first color to source
    colorArr[0] = 1

    # Create a queue (FIFO) of vertex numbers and enqueue source vertex for BFS
    # traversal
    queue = []
    queue.append(0)

    # Run while there are vertices in queue (Similar to BFS)
    while queue:

        u = queue.pop()

        # Return false if there is a self-loop
        if A[u, u] == 1:
            return False

        for v in range(n):

            # An edge from u to v exists and destination
            # v is not colored
            if A[u, v] == 1 and colorArr[v] == -1:

                # Assign alternate color to this
                # adjacent v of u
                colorArr[v] = 1 - colorArr[u]
                queue.append(v)

            # An edge from u to v exists and destination
            # v is colored with same color as u
            elif A[u, v] == 1 and colorArr[v] == colorArr[u]:
                return False

    # If we reach here, then all adjacent
    # vertices can be colored with alternate
    # color
    return True


def get_corr_spin_p(X, Y, spins):
    '''
    Function to compute the p-value of a correlation score compared to spun
    distributions (for X)
    '''

    N_nodes, N_spins = spins.shape
    emp_corr, _ = pearsonr(X, Y)
    spin_corr = np.zeros((N_spins))
    for i in range(N_spins):
        spin_corr[i], _ = pearsonr(X[spins[:, i]], Y)
    p_spin = get_p_value(spin_corr, emp_corr)

    return p_spin


def get_p_value(perm, emp):
    '''
    Utility function to compute the p-value of a score, relative to a null
    distribution

    Parameters
    ----------
    perm: array-like
        Null distribution of (permuted) scores.
    emp: float
        Empirical score.
    '''

    k = len(perm)
    return len(np.where(abs(perm-np.mean(perm)) > abs(emp-np.mean(perm)))[0])/k


def standardize_scores(surr, emp, axis=None, ignore_nan=False):
    '''
    Utility function to standardize a score relative to a null distribution.

    Parameters
    ----------
    perm: array-like
        Null distribution of scores.
    emp: float
        Empirical score.
    '''
    if ignore_nan:
        return (emp - np.nanmean(surr, axis=axis)) / np.nanstd(surr, axis=axis)
    else:
        return (emp - surr.mean(axis=axis)) / surr.std(axis=axis)


def load_pickle(path):

    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    return data


def save_pickle(data, path):

    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def split_matrix_hemi(A, parc):
    '''
    Function to split the inter-hemispheric data of a matrix.
    '''

    node_mask, hemi_mask, _ = load_data.get_node_masks(
        None, N_hemi='both', N_parcel=parc)
    hemi_labels = hemi_mask[node_mask]

    A_left = A[hemi_labels == 'L', :][:, hemi_labels == 'L']
    A_right = A[hemi_labels == 'R', :][:, hemi_labels == 'R']

    return A_left, A_right


def multi_pearsonr(X, Y):
    '''
    Function to compute the pearson's correlation between a distribution `X`
    and a large number of distributions, stored in the matrix `Y`.

    Parameters
    ----------
    X : (n,) ndarray
        Vector of n observation.
    Y: (n, k) ndarray
        Matrix, where each column is a vector of correlation that we want to
        correlate with X.

    Returns
    -------
    R : (k,) ndarray
        Vector of Pearson's r scores representing the correlation between the
        independent variable and all the dependent variables
    '''

    n, k = Y.shape
    X_z = (X - X.mean()) / X.std()
    Y_z = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    X_z_repeat = np.repeat(X_z[:, np.newaxis], k, axis=1)
    R = np.mean(X_z_repeat * Y_z, axis=0)

    return R