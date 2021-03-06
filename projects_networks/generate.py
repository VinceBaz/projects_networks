# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:58:26 2020

@author: Vincent Bazinet
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from . import load_data as ld
from mapalign.embed import compute_diffusion_map


def generate_network(network_type, lattice_kws=None, ER_kws=None):
    '''
    Function to generate networks.

    Parameters
    ----------
    network_type : str
        Name of the network that we wish to generate. The available options are
        'lattice' and 'weighted_ER'.
    '''
    Network = {}

    if network_type == 'lattice':

        n_rows = 10
        gaps = 1

        if lattice_kws is not None:
            if "n_rows" in lattice_kws:
                n_rows = lattice_kws['n_rows']
            if "gaps" in lattice_kws:
                gaps = lattice_kws['gaps']

        Network['adj'], Network['coords'] = generate_lattice(n_rows, gaps)
        Network['dist'] = cdist(Network['coords'], Network['coords'])

    if network_type == 'weighted_ER':

        n = 50
        p = 0.20

        if ER_kws is not None:
            if 'n' in ER_kws:
                n = ER_kws['n']
            if 'p' in ER_kws:
                p = ER_kws['p']

        Network['adj'] = generate_weighted_ER(n, p)

    if network_type == 'binary_ER':

        n = 50
        p = 0.20

        if ER_kws is not None:
            if 'n' in ER_kws:
                n = ER_kws['n']
            if 'p' in ER_kws:
                p = ER_kws['p']

        Network['adj'] = generate_binary_ER(n, p)

    Network['str'] = np.sum(Network['adj'], axis=0)
    Network['PCs'] = ld.getPCs(Network['adj'])[0]
    de = compute_diffusion_map(Network['adj'],
                               n_components=10,
                               return_result=True)
    Network["de"] = de[0]

    return Network


def generate_weighted_ER(n, p):

    # Get the binary ER network
    ER = nx.to_numpy_array(nx.generators.random_graphs.gnp_random_graph(n, p))

    # Only keep the upper triangle of the matrix
    upper_ER = np.triu(ER)

    weighted_ER = ER.copy()
    N_edges = np.count_nonzero(upper_ER)
    weights = np.random.rand(N_edges)
    upper_ER[upper_ER > 0] = weights

    weighted_ER = upper_ER + upper_ER.T

    return weighted_ER


def generate_binary_ER(n, p):

    # Get the binary ER network
    ER = nx.to_numpy_array(nx.generators.random_graphs.gnp_random_graph(n, p))

    return ER


def generate_lattice(n_rows, gaps):

    adjacency = np.zeros((n_rows ** 2, n_rows ** 2))
    coords = np.indices((n_rows, n_rows)).reshape(2, -1).T
    dists = cdist(coords, coords, metric="chebyshev")

    # link together all the rows, according to the size of the gaps
    for i in range(0, n_rows, gaps):
        row_id = np.where(coords[:, 0] == i)[0]
        row_adj = adjacency[:, row_id][row_id, :]
        row_adj[np.where(dists[:, row_id][row_id, :] == 1)] = 1

        maskA = np.zeros((n_rows ** 2, n_rows ** 2), dtype=bool)
        maskB = np.zeros((n_rows ** 2, n_rows ** 2), dtype=bool)
        maskA[:, row_id] = True
        maskB[row_id, :] = True
        mask = np.zeros((n_rows ** 2, n_rows ** 2), dtype=bool)
        mask[(maskA) & (maskB)] = True

        adjacency[mask] = row_adj.reshape(-1)

    # link together all the columns, according to the size of the gaps
    for i in range(0, n_rows, gaps):
        col_id = np.where(coords[:, 1] == i)[0]
        col_adj = adjacency[:, col_id][col_id, :]
        col_adj[np.where(dists[:, col_id][col_id, :] == 1)] = 1

        maskA = np.zeros((n_rows ** 2, n_rows ** 2), dtype=bool)
        maskB = np.zeros((n_rows ** 2, n_rows ** 2), dtype=bool)
        maskA[:, col_id] = True
        maskB[col_id, :] = True
        mask = np.zeros((n_rows ** 2, n_rows ** 2), dtype=bool)
        mask[(maskA) & (maskB)] = True

        adjacency[mask] = col_adj.reshape(-1)

    deg = np.sum(adjacency, axis=0)
    adjacency = np.delete(adjacency, np.where(deg == 0)[0], axis=0)
    adjacency = np.delete(adjacency, np.where(deg == 0)[0], axis=1)
    coords = np.delete(coords, np.where(deg == 0)[0], axis=0)

    return adjacency, coords
