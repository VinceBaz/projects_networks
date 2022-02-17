import random
import numpy as np
from . import load_data
from neuromaps.nulls import alexander_bloch
from neuromaps.images import annot_to_gifti


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

    # Gather information about the network
    N_nodes = len(A)
    sum_weights = np.sum(A, axis=None)

    # Gather information about the network's edges
    edges = np.array(np.where(A > 0)).T.tolist()
    N_edges = len(edges)

    # Normalize network weights
    A = A / sum_weights

    # Compute global assortativity
    k_norm = np.sum(A, axis=0)
    M_mean = np.sum(k_norm * M)
    M_sd = np.sqrt(np.sum(k_norm * ((M-M_mean)**2)))
    zM = (M - M_mean) / M_sd
    zj = np.repeat(zM[np.newaxis, :], N_nodes, axis=0)
    zi = zj.T
    assort = (A * zi * zj).sum()

    # while assort is less than goal assort
    it = 0
    diff = abs(assort - g)
    while diff > epsilon:

        it += 1

        # Choose 2 edges at random
        rand1 = random.randrange(0, N_edges)
        rand2 = random.randrange(0, N_edges)
        e1 = [edges[rand1][0], edges[rand1][1]]
        e2 = [edges[rand2][0], edges[rand2][1]]

        # if swap creates a self-loop or a multiedge, resample
        if e1[0] == e2[1] or e2[0] == e1[1]:
            continue
        elif A[e1[0], e2[1]] > 0 or A[e2[0], e1[1]] > 0:
            continue
        else:
            # swap edges
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

            # Compute updated assortativity
            k_norm = np.sum(A, axis=0)
            M_mean = np.sum(k_norm * M)
            M_sd = np.sqrt(np.sum(k_norm * ((M-M_mean)**2)))
            zM = (M - M_mean) / M_sd
            zj = np.repeat(zM[np.newaxis, :], N_nodes, axis=0)
            zi = zj.T
            assortNew = (A * zi * zj).sum()

            diffNew = abs(assortNew - g)

            # If swapped network has larger assortativity, keep swap
            if diffNew < diff:

                # Update assortativity and diff
                assort = assortNew
                diff = diffNew

                # Update edge information
                if und:
                    edges[rand1] = [e1[0], e2[1]]
                    edges[rand2] = [e2[0], e1[1]]

                    edges.remove([e1[1], e1[0]])
                    edges.remove([e2[1], e2[0]])
                    edges.append([e2[1], e1[0]])
                    edges.append([e1[1], e2[0]])
                else:
                    edges[rand1] = [e1[0], e2[1]]
                    edges[rand2] = [e2[0], e1[1]]

            # Else, swap back
            else:
                if und:
                    A[e1[0], e1[1]] = w1
                    A[e1[1], e1[0]] = w1
                    A[e2[0], e2[1]] = w2
                    A[e2[1], e2[0]] = w2
                    A[e1[0], e2[1]] = 0
                    A[e2[1], e1[0]] = 0
                    A[e2[0], e1[1]] = 0
                    A[e1[1], e2[0]] = 0
                else:
                    A[e1[0], e1[1]] = w1
                    A[e2[0], e2[1]] = w2
                    A[e1[0], e2[1]] = 0
                    A[e2[0], e1[1]] = 0

    # Unnormalize the weights
    A = A * sum_weights

    return A, it


def generate_spins(parcel, info_path, hemi='', k=10000):
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
    hemi_info = load_data._load_hemi_info(parcel, info_path)
    sub_info = load_data._load_subcortex_info(parcel, info_path)

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
