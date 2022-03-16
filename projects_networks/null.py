import random
import numpy as np
from . import load_data
from neuromaps.nulls import alexander_bloch
from neuromaps.images import annot_to_gifti
from projects_networks.assortativity import weighted_assort


def assort_preserv_swap(A, M, g, epsilon=0.0001, und=True):
    '''
    Function to generate surrogate networks with preserved global assortativity
    and degree.

    Parameters
    ----------
    A : (N_nodes, N_nodes) ndarray
        Adjacency matrix of the network.
    M : (N_nodes,) ndarray
        Vector of node attributes.
    g : float
        Global assortativity score that we want (empirical).
    epsilon : float
        Error term when calculating the difference between the empirical
        and the permuted assortativity.
    und: Boolean
        Specifies whether the network is undirected (True) or directed (False).
    '''

    # Get random edges
    def get_random_edge(edges, N_edges):
        not_found = True
        while not_found:

            not_found = False
            rand1 = random.randrange(0, N_edges)
            rand2 = random.randrange(0, N_edges)
            e1 = [edges[rand1][0], edges[rand1][1]]
            e2 = [edges[rand2][0], edges[rand2][1]]

            # if swap creates a self-loop or a multiedge, resample
            if e1[0] == e2[1] or e2[0] == e1[1]:
                not_found = True
            elif A[e1[0], e2[1]] > 0 or A[e2[0], e1[1]] > 0:
                not_found = True

        return rand1, rand2, e1, e2

    # Swap edges
    def swap_edges(A, e1, e2):
        w1 = A[e1[0], e1[1]]
        w2 = A[e2[0], e2[1]]
        if und:
            A[e1[0], e1[1]] = 0
            A[e1[1], e1[0]] = 0
            A[e2[0], e2[1]] = 0
            A[e2[1], e2[0]] = 0
            A[e1[0], e2[1]] = w1
            A[e2[1], e1[0]] = w1
            A[e2[0], e1[1]] = w2
            A[e1[1], e2[0]] = w2
        else:
            A[e1[0], e1[1]] = 0
            A[e2[0], e2[1]] = 0
            A[e1[0], e2[1]] = w1
            A[e2[0], e1[1]] = w2

    # Update edge information
    def update_edges(edges, rand1, rand2, e1, e2):
        if und:
            edges[rand1] = [e1[0], e2[1]]
            edges[rand2] = [e2[0], e1[1]]

            id1 = edges.index([e1[1], e1[0]])
            id2 = edges.index([e2[1], e2[0]])
            edges[id1] = [e2[1], e1[0]]
            edges[id2] = [e1[1], e2[0]]
        else:
            edges[rand1] = [e1[0], e2[1]]
            edges[rand2] = [e2[0], e1[1]]

    # Gather information about the network
    sum_weights = np.sum(A, axis=None)

    # Gather information about the network's edges
    edges = np.array(np.where(A > 0)).T.tolist()
    N_edges = len(edges)

    # Normalize network weights
    A = A / sum_weights

    # diff with goal assort
    diff = abs(weighted_assort(A, M, directed=(not und), normalize=False) - g)

    # while diff with goal assort is larger than epsilon
    it = 0
    while diff > epsilon:

        it += 1

        # Choose 2 edges at random
        rand1, rand2, e1, e2 = get_random_edge(edges, N_edges)
        swap_edges(A, e1, e2)

        diff_new = abs(weighted_assort(A, M, directed=(not und),
                                       normalize=False
                                       ) - g)

        # If swapped network has larger assortativity, keep swap
        if diff_new < diff:
            diff = diff_new
            update_edges(edges, rand1, rand2, e1, e2)

        # Else, swap back
        else:
            swap_edges(A, [e1[0], e2[1]], [e2[0], e1[1]])

    # Unnormalize the weights
    A = A * sum_weights

    return A, it


def generate_spins(parcel, hemi='', k=10000):
    '''
    Function to generate spun permutation of a parcellation's parcels.
    '''

    # Load information about the parcellation
    parcel_info = load_data.get_general_parcellation_info(parcel)
    order = parcel_info[0]
    lhannot = parcel_info[2]
    rhannot = parcel_info[3]

    # Generate the nulls
    spins_LR = alexander_bloch(None,
                               'fsaverage',
                               density='164k',
                               n_perm=10000,
                               parcellation=annot_to_gifti((lhannot, rhannot)),
                               seed=1234)

    # Get some info about hemispheres and subcortex
    hemi_info = load_data._load_hemi_info(parcel)
    sub_info = load_data._load_subcortex_info(parcel)

    # Remove subcortex info from hemi_info (spins are only on the surface)
    hemi_info = np.delete(hemi_info, sub_info)

    _, (n_R, n_L) = np.unique(hemi_info, return_counts=True)

    # If order is RL, we must invert the order of the spun annotations
    if order == 'RL':
        spins = np.zeros((spins_LR.shape))
        spins[:n_R, :] = spins_LR[n_L:, :] - n_L
        spins[n_R:, :] = spins_LR[:n_L, :] + n_R
    elif order == 'LR':
        spins = spins_LR

    # Only keep the information about the hemisphere we want
    if hemi == "L":
        if order == 'RL':
            spins = spins[n_R:, :] - n_R
        elif order == 'LR':
            spins = spins[:n_L, :]
    elif hemi == 'R':
        if order == 'RL':
            spins = spins[:n_R, :]
        elif order == 'LR':
            spins = spins[n_L:, :] - n_L

    return spins.astype(int)
