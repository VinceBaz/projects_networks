# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:58:26 2020

This file contains function to generate different types of network.

@author: Vincent Bazinet
"""

import numpy as np
import bct
import networkx as nx
from fa2 import ForceAtlas2
from scipy.spatial.distance import cdist
from . import load_data as ld
from mapalign.embed import compute_diffusion_map
from netneurotools.metrics import communicability_wei
import warnings


def generate_network(network_type, lattice_kws=None, ER_kws=None,
                     custom_kws=None, ignore=None):
    '''
    Function to generate networks.

    Parameters
    ----------
    network_type : str
        Name of the network that we wish to generate. Currently, the
        available options are:
            'lattice'
            'weighted_ER'
            'binary_ER'
            'assortative_toy'
            'dissassortative_toy'
            'custom'
    ignore : List
        List of dictionary entries that are to be ignored when loading
        the results. Available options are: 'sp' (shortest path).
    Returns
    -------
    Network : dict
        Dictionary storing relevant information about the generated network.
    '''
    Network = {}

    if ignore is None:
        ignore = []

    # Part 1: Adjacency matrix and coordinates (when possible)
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
    if network_type == 'assortative_toy':
        Network['adj'] = generate_assortative_toy()
    if network_type == 'disassortative_toy':
        Network['adj'] = generate_disassortative_toy()
    if network_type == 'custom':
        Network['adj'] = custom_kws['adj']

    # Test whether the network is connected. Raise a warning if not...
    if not np.all(bct.reachdist(Network['adj'])[0]):
        warnings.warn(("This brain network appears to be disconnected. This "
                       "might cause problem for the computation of the other "
                       "measures"))

    # Part 2: Topological measures
    Network['str'] = Network['adj'].sum(axis=0)
    Network['cc'] = bct.clustering_coef_wu(Network['adj'])
    Network['subc'] = bct.subgraph_centrality(Network['adj'])
    Network['ec'] = bct.eigenvector_centrality_und(Network['adj'])
    Network['com'] = communicability_wei(Network['adj'])

    if 'sp' not in ignore:
        Network['sp'], _ = bct.distance_wei(Network['adj'])
        Network['msp'] = Network['sp'].mean(axis=0)
        Network['r_eff'] = 1/Network['msp']

    # Part 3: Embeddings
    Network['PCs'] = ld.getPCs(Network['adj'])[0]
    de = compute_diffusion_map(Network['adj'],
                               n_components=10,
                               return_result=True,
                               skip_checks=True)
    Network["de"] = de[0]
    Network['de_extra'] = de[1]
    Network['fa'] = _forceAtlasEmbedding(Network, verbose=False)

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


def generate_assortative_toy():

    adjacency = np.array([[0,1,1,1,1,1,1,1,0,0,0,0,0,0],
                          [1,0,1,0,0,0,1,0,0,0,0,0,0,0],
                          [1,1,0,1,0,0,0,0,0,0,0,0,0,0],
                          [1,0,1,0,1,0,0,0,0,0,0,0,0,0],
                          [1,0,0,1,0,1,0,0,0,0,0,0,0,0],
                          [1,0,0,0,1,0,1,0,0,0,0,0,0,0],
                          [1,1,0,0,0,1,0,0,0,0,0,0,0,0],
                          [1,0,0,0,0,0,0,0,1,1,1,1,1,1],
                          [0,0,0,0,0,0,0,1,0,1,0,0,0,1],
                          [0,0,0,0,0,0,0,1,1,0,1,0,0,0],
                          [0,0,0,0,0,0,0,1,0,1,0,1,0,0],
                          [0,0,0,0,0,0,0,1,0,0,1,0,1,0],
                          [0,0,0,0,0,0,0,1,0,0,0,1,0,1],
                          [0,0,0,0,0,0,0,1,1,0,0,0,1,0]])

    return adjacency


def generate_disassortative_toy():

    adjacency = np.array([[0,1,1,1,1,1,1,0,0,0,0,0,0,0],
                          [1,0,1,0,0,0,1,0,1,0,0,0,0,0],
                          [1,1,0,1,0,0,0,0,0,0,0,0,0,0],
                          [1,0,1,0,1,0,0,0,0,0,0,0,0,0],
                          [1,0,0,1,0,1,0,0,0,0,0,0,0,0],
                          [1,0,0,0,1,0,1,0,0,0,0,0,0,0],
                          [1,1,0,0,0,1,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,1,1,1,1,1,1],
                          [0,1,0,0,0,0,0,1,0,1,0,0,0,1],
                          [0,0,0,0,0,0,0,1,1,0,1,0,0,0],
                          [0,0,0,0,0,0,0,1,0,1,0,1,0,0],
                          [0,0,0,0,0,0,0,1,0,0,1,0,1,0],
                          [0,0,0,0,0,0,0,1,0,0,0,1,0,1],
                          [0,0,0,0,0,0,0,1,1,0,0,0,1,0]])

    return adjacency


def _forceAtlasEmbedding(Network, verbose=True, dissuade_hubs=True,
                         scalingRatio=2.0):

    adj = Network["adj"]

    forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=dissuade_hubs,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=5.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=scalingRatio,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=verbose)

    positions = forceatlas2.forceatlas2(adj, pos=None, iterations=2000)
    position_array = np.array(positions)

    return position_array
