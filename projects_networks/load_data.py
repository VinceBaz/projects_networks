# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:12:05 2020

Code to load brain network data from a "data" folder stored locally
into an easy to use python dictionary.

@author: Vincent Bazinet
"""

import numpy as np
from scipy.spatial.distance import cdist
import bct
from mapalign.embed import compute_diffusion_map
import os
import abagen
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from . import null


def load_network(kind, parcel, data="lau", hemi="both", binary=False,
                 version=1, subset="all", path=None):
    '''
    Function to load a network as well as its attributes

    Parameters
    ----------
    kind : string
        Either 'SC' or 'FC'.
    hemi : string
        Either "both", "L" or "R".
    data : string
        Either "HCP" or "lau".
    parcel : string
        Either "68", "114", ... [if 'lau'] / "s400", "s800" [if "HCP"]
    version : int
        Either 1 (consensus computed without subcortex) or 2 (consensus
        conputed with subcortex)
    subset : string
        Either 'discov', 'valid' or 'all'
    path : string
        path to the "data" folder in which the data will be stored. If
        none, then assumes that path is current folder.

    Returns
    -------
    Network : dictionary
        Dictionary storing relevant attributes about the network
    '''

    # Initialize dictionary + store basic information about the network
    Network = {}
    Network["info"] = {}
    Network["info"]["kind"] = kind
    Network["info"]["parcel"] = parcel
    Network["info"]["data"] = data
    Network["info"]["hemi"] = hemi
    Network["info"]["binary"] = binary
    Network["info"]["version"] = version
    Network["info"]["subset"] = subset

    # Modify parameter names to what they are in file names
    version = '' if version == 1 else "_v2"
    binary = 'b' if binary else ''
    subset = '' if subset == 'all' else subset
    hemi = '' if hemi == 'both' else hemi

    # Store important paths for loading the relevant data
    mainPath = path+"/brainNetworks/"+data+"/"
    matrixPath = mainPath+"matrices/"+subset+kind+parcel+hemi+binary+version

    # Store general information about the network's parcellation
    parcel_info = get_general_parcellation_info(parcel)
    Network['order'] = parcel_info[0]
    Network['noplot'] = parcel_info[1]
    Network['lhannot'] = parcel_info[2]
    Network['rhannot'] = parcel_info[3]

    # Load the cammoun_id of the parcellation, if Cammoun (i.e. 033, 060, etc.)
    if parcel[0] != 's':
        Network['cammoun_id'] = parcel_to_n(parcel)

    # masks
    masks = get_node_masks(Network, path=mainPath)
    Network['node_mask'] = masks[0]
    Network['hemi_mask'] = masks[1]
    Network['subcortex_mask'] = masks[2]

    # hemisphere
    Network['hemi'] = get_hemi(Network, path=mainPath)

    # coordinates
    Network['coords'] = get_coordinates(Network, path=mainPath)

    # Adjacency matrix
    Network['adj'], last_modified = get_adjacency(Network, matrixPath,
                                                  minimal_processing=True,
                                                  return_last_modified=True)

    # node strength
    Network["str"] = np.sum(Network['adj'], axis=0)

    # Inverse of adjacency matrix
    inv = Network['adj'].copy()
    inv[Network['adj'] > 0] = 1/inv[Network['adj'] > 0]
    Network["inv_adj"] = inv

    # distance
    Network["dist"] = cdist(Network["coords"], Network["coords"])

    # clustering coefficient
    Network["cc"] = bct.clustering_coef_wu(Network['adj'])

    # shortest path
    #
    # Loaded from saved file...
    # IF file not found OR Adjacency was modified after creation,
    # then recompute measure
    path = matrixPath+"/sp.npy"

    if not os.path.exists(path):

        print("shortest path not found")
        print("computing shortest path...")

        Network["sp"] = bct.distance_wei(Network["inv_adj"])[0]
        np.save(matrixPath+"/sp.npy", Network["sp"])

    elif os.path.getmtime(path) < last_modified:

        print("new adjacency matrix was found")
        print("computing shortest paths...")

        Network["sp"] = bct.distance_wei(Network["inv_adj"])[0]
        np.save(matrixPath+"/sp.npy", Network["sp"])

    else:

        Network["sp"] = np.load(path)

    # diffusion embedding
    de = compute_diffusion_map(Network['adj'],
                               n_components=10,
                               return_result=True)
    Network["de"] = de[0]
    Network["de_extra"] = de[1]

    # Principal components
    Network['PCs'], Network['PCs_ev'] = getPCs(Network['adj'])

    # eigenvector centrality
    Network["ec"] = bct.eigenvector_centrality_und(Network['adj'])

    # mean first passage time
    Network["mfpt"] = bct.mean_first_passage_time(Network['adj'])

    # betweenness centrality
    #
    # Loaded from saved file...
    # IF file not found OR Adjacency was modified after creation,
    # then recompute measure
    path = matrixPath+"/bc.npy"
    if not os.path.exists(path):

        print("betweenness centrality not found")
        print("computing betweenness centrality...")
        Network["bc"] = bct.betweenness_wei(Network["inv_adj"])
        np.save(matrixPath+"/bc.npy", Network["bc"])

    elif os.path.getmtime(path) < last_modified:
        print("new adjacency matrix was found")
        print("recomputing betweeness centrality...")
        Network["bc"] = bct.betweenness_wei(Network["inv_adj"])
        np.save(matrixPath+"/bc.npy", Network["bc"])

    else:
        Network["bc"] = np.load(path)

    # routing efficiency
    Network["r_eff"] = efficiency(Network)

    # diffusion efficiency
    Network["d_eff"] = efficiency_diffusion(Network)

    # subgraph centrality
    Network["subc"] = bct.subgraph_centrality(Network["adj"])

    # closeness centrality
    Network['clo'] = 1/np.mean(Network['sp'], axis=0)

    # communities + participation coefficient
    path = matrixPath+"/communities/"
    if os.path.exists(path):
        files = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path, i)) and 'ci_' in i:
                files.append(i)
        if len(files) > 0:
            Network["ci"] = []
            for file in files:
                Network["ci"].append(np.load(os.path.join(path, file)))

            Network["ppc"] = []
            for i in range(len(files)):
                ppc = bct.participation_coef(Network['adj'], Network["ci"][i])
                Network["ppc"].append(ppc)

    if (data == "HCP") and (kind == "SC"):
        path = mainPath+"matrices/"+subset+kind+parcel+hemi+"_lengths.npy"
        if os.path.exists(path):
            Network["lengths"] = np.load(path)

    # streamline connection lengths
    path = matrixPath+"/len.npy"
    if os.path.exists(path):
        Network['len'] = np.load(path)

    # ROI names
    if parcel[0] != "s":
        Network['ROInames'] = get_ROInames(Network)

    # geodesic distances between nodes
    if parcel[0] == "s":
        n = parcel[1:]
        fname_l = n + "Parcels7Networks_lh_dist.csv"
        fname_r = n + "Parcels7Networks_lh_dist.csv"
    else:
        fname_l = "scale" + Network['cammoun_id'] + "_lh_dist.csv"
        fname_r = "scale" + Network['cammoun_id'] + "_rh_dist.csv"
    Network['geo_dist_L'] = pd.read_csv(mainPath+"/geodesic/medial/" + fname_l,
                                        header=None).values
    Network['geo_dist_R'] = pd.read_csv(mainPath+"/geodesic/medial/" + fname_r,
                                        header=None).values

    return Network


def load_genes(parcel, data="lau", hemi="both", path=None, PC_scaled=True):
    '''
    Load gene-related information
    '''
    mainPath = path+"/GeneExpression/"

    # Dictionary storing gene information for the network
    G = {}

    if hemi == "both":
        hemi = ''

    # Gene names
    path = mainPath+"name_genes_"+parcel+hemi+".npy"
    if os.path.exists(path):
        # (all)
        G["names"] = np.load(path, allow_pickle=True).tolist()
        # (brain)
        G["names_brain"] = abagen.fetch_gene_group('brain')
        geneBrainsID = np.isin(G["names"], G["names_brain"])
        G["names_brain"] = np.array(G["names"])[geneBrainsID].tolist()

    # Gene expression
    path = mainPath+"gene_"+parcel+hemi+".npy"
    if os.path.exists(path):
        # (all)
        G["ex"] = np.load(path)
        # (brain)
        G["ex_brain"] = G["ex"][:, geneBrainsID]

    # Principal components
    if "ex" in G:
        # (all)
        G["PCs"], G["PC_evs"] = getPCAgene(G["ex"],
                                           scaled=PC_scaled)
        # (brain)
        G["PCs_brain"], G["PC_evs_brain"] = getPCAgene(G["ex_brain"],
                                                       scaled=PC_scaled)

    # Differential stability
    path = mainPath+"DS_"+parcel+hemi+".npy"
    if os.path.exists(path):
        # (all)
        G['DS'] = np.load(path)
        # (brain)
        G["DS_brain"] = G['DS'][geneBrainsID]

    return G


def load_annotations(parcel, data="lau", hemi="both",
                     subset="all", path=None):
    '''
    Load non-topological annotations as well as some tools useful
    for investigating these annotations (e.g. spin samples)
    '''

    mainPath = path+"/brainNetworks/"+data+"/"
    info_path = mainPath + "/matrices/general_info"
    ANN = {}

    if subset == "all":
        subset = ''

    if hemi == "both":
        hemi = ''

    # RSN
    path = mainPath+"annotations/RSN"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["RSN"] = np.load(path)

    # RSN names
    path = mainPath+"annotations/RSN_names.npy"
    if os.path.exists(path):
        ANN["RSN_names"] = np.load(path).tolist()

    # VonEconomo type
    path = mainPath+"annotations/VE"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["ve"] = np.load(path)

    # Von Economo names
    path = mainPath+"annotations/VE_names.npy"
    if os.path.exists(path):
        ANN["ve_names"] = np.load(path).tolist()

    # Spun permutation
    path = mainPath+"surrogates/spins_"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["spin"] = np.load(path)
    else:
        print("spun permutation not found...")
        print("computing and saving...")
        order, _, lhannot, rhannot = get_general_parcellation_info(parcel)
        ANN['spin'] = null.generate_spins(parcel, lhannot, rhannot,
                                          order, info_path, hemi=hemi)
        np.save(path, ANN['spin'])

    # Morphometric property (T1w/T2w)
    path = mainPath+"annotations/T1wT2w/"+subset+"_"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["t1t2"] = np.mean(np.load(path), axis=0)

    # Morphometric property (Thickness)
    path = mainPath+"annotations/thi/"+subset+"_"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["thi"] = np.mean(np.load(path), axis=0)

    # Functional activation matrix
    path = mainPath+"../../neurosynth/parcellated/"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["fa"] = np.load(path)

        # Functional activation PCs
        ANN['fa_PCs'], ANN['fa_PCs_ev'] = getPCAgene(ANN["fa"], scaled=True)

    return ANN


def parcel_to_n(parcel):

    mapping = {}
    mapping["68"] = "033"
    mapping['83'] = "033"
    mapping["114"] = "060"
    mapping['129'] = '060'
    mapping["219"] = "125"
    mapping['234'] = '125'
    mapping["448"] = "250"
    mapping['463'] = '250'
    mapping["1000"] = "500"
    mapping['1015'] = '500'

    return mapping[parcel]


def get_adjacency(Network, matrix_path, minimal_processing=True,
                  return_last_modified=True):

    path = matrix_path + ".npy"

    # Load adjacency matrix
    A = np.load(path)

    # Store time when last modified
    last_modified = os.path.getmtime(path)

    if minimal_processing:
        # set negative values to 0, fill diagonal, make symmetric
        A[A < 0] = 0
        np.fill_diagonal(A, 0)
        A = (A + A.T)/2

    if return_last_modified:
        return A, last_modified
    else:
        return A


def get_coordinates(Network, path='../data/brainNetworks/lau'):
    '''
    Function to get the coordinates of the nodes in the network.
    '''

    # Get node mask of network
    mask = Network['node_mask']

    # Get some information about the network of interest (HCP or lau)
    network_data = Network['info']['data']

    # Store path to the .npy coordinate file
    if network_data == 'lau':
        path = path + "coords/coords" + Network['cammoun_id'] + '.npy'
    elif network_data == 'HCP':
        path = path + "coords/coords" + Network['info']['parcel'] + '.npy'

    coords = np.load(path)

    return coords[mask, :]


def get_general_parcellation_info(parcel):
    '''
    Function to get general information about the parcellation.
    '''

    home = os.path.expanduser("~")

    if parcel[0] == "s":
        n = parcel[1:]
        order = "LR"
        noplot = [b'Background+FreeSurfer_Defined_Medial_Wall',
                  b'']
        lhannot = (home+"/"
                   "nnt-data/"
                   "atl-schaefer2018/"
                   "fsaverage/"
                   "atl-Schaefer2018_space-fsaverage_"
                   "hemi-L_desc-"+n+"Parcels7Networks_"
                   "deterministic.annot")
        rhannot = (home+"/"
                   "nnt-data/"
                   "atl-schaefer2018/"
                   "fsaverage/"
                   "atl-Schaefer2018_space-fsaverage_"
                   "hemi-R_desc-"+n+"Parcels7Networks_"
                   "deterministic.annot")
    else:
        n = parcel_to_n(parcel)
        order = "RL"
        noplot = None
        lhannot = (home+"/"
                   "nnt-data/"
                   "atl-cammoun2012/"
                   "fsaverage/"
                   "atl-Cammoun2012_space-fsaverage_"
                   "res-"+n+"_hemi-L_deterministic.annot")
        rhannot = (home+"/"
                   "nnt-data/"
                   "atl-cammoun2012/"
                   "fsaverage/"
                   "atl-Cammoun2012_space-fsaverage_"
                   "res-"+n+"_hemi-R_deterministic.annot")

    return order, noplot, lhannot, rhannot


def get_node_masks(N, path="../data/brainNetworks/lau"):
    '''
    Function to get a mask of the nodes of this particular network (1), given
    the original parcellation (0).
    '''

    # Load info about the network
    network_hemi = N['info']['hemi']
    network_data = N['info']['data']
    network_parcel = N['info']['parcel']

    # Load general info about hemispheres and subcortex
    info_path = path + "/matrices/general_info"
    hemi_mask = _load_hemi_info(network_parcel, info_path)
    subcortex_nodes = _load_subcortex_info(network_parcel, info_path)

    # Initialize node mask
    n = len(hemi_mask)
    node_mask = np.zeros((n), dtype=bool)

    # Get subcortex mask
    subcortex_mask = np.zeros((n), dtype=bool)
    subcortex_mask[subcortex_nodes] = True

    # Figure out whether this parcellation contains the subcortex
    subcortex = False
    if network_data == 'lau':
        if N["info"]["parcel"] in ['83', '129', '234', '463', '1015']:
            subcortex = True

    # Get a list of indices of the nodes in the network
    if not subcortex:

        if network_hemi == 'L':
            nodes = np.where((hemi_mask == 1) & (subcortex_mask == 0))[0]

        elif network_hemi == 'R':
            nodes = np.where((hemi_mask == 0) & (subcortex_mask == 0))[0]

        elif network_hemi == 'both':
            nodes = np.where((subcortex_mask == 0))[0]

    elif subcortex:

        if network_hemi == 'L':
            nodes = np.where(hemi_mask == 1)[0]

        elif network_hemi == 'R':
            nodes = np.where(hemi_mask == 0)[0]

        elif network_hemi == 'both':
            nodes = np.ones((n), dtype=bool)

    node_mask[nodes] = True

    return node_mask, hemi_mask, subcortex_mask


def get_hemi(Network, path="../data/brainNetworks/lau"):
    '''
    Function to get hemisphere information for the nodes in the network.
    This hemispheric information is viewed as an annotation. In other words
    the information is only given for the nodes of the network (so we rely on
    our network's 'node_mask' to extract the nodes of interest)
    '''
    # Get node mask of network
    mask = Network['node_mask']

    # Get some information about the network of interest (HCP or lau)
    network_parcel = Network['info']['parcel']

    # Get information about hemispheres for the given parcellation
    info_path = path + "/matrices/general_info"
    hemi = _load_hemi_info(network_parcel, info_path)

    return hemi[mask]


def get_streamline_length(Network, path='../data'):
    '''
    Function to get the streamline lengths of a structural consensus network.
    '''

    matricesPath = path+"/brainNetworks/lau/matrices"
    network_hemi = Network['info']['hemi']

    with open(matricesPath+"/general_info/hemi.pkl", "rb") as handle:
        hemi = pickle.load(handle)
    hemi = hemi[Network['cammoun_id']].reshape(-1)

    with open(matricesPath+"/general_info/subcortexNodes.pkl", "rb") as handle:
        subcortexNodes = pickle.load(handle)
    subcortexNodes = subcortexNodes[Network['cammoun_id']]
    node_type = np.zeros(len(hemi))
    node_type[subcortexNodes] = 1

    if network_hemi == "L":
        ignored = np.where((hemi == 0) | (node_type == 1))[0]
    elif network_hemi == 'R':
        ignored = np.where((hemi == 1) or (node_type == 1))[0]
    else:
        ignored = np.where((node_type == 1))[0]

    indSC = np.load((path +
                     "/Lausanne/struct/struct_len_scale" +
                     Network['cammoun_id'] +
                     ".npy")
                    )

    indSC = np.delete(indSC, (7, 12, 43), axis=2)
    indSC = np.delete(indSC, ignored, axis=0)
    indSC = np.delete(indSC, ignored, axis=1)
    indSC[indSC == 0] = np.nan

    SC_len = Network['adj'].copy()
    SC_len[SC_len > 0] = np.nanmean(indSC, axis=2)[SC_len > 0]

    return SC_len


def efficiency(Network):

    sp = Network["sp"]
    n = len(sp)

    efficiency_local = np.zeros((n))
    for i in range(n):
        efficiency_local[i] = np.mean(1/np.delete(sp[i, :], i))
    efficiency_global = np.mean(efficiency_local)

    return efficiency_global


def efficiency_diffusion(Network):

    if isinstance(Network, dict):
        mfpt = Network["mfpt"]
    else:
        mfpt = bct.mean_first_passage_time(Network)
    n = len(mfpt)

    efficiency_local = np.zeros((n))
    for i in range(n):
        efficiency_local[i] = np.mean(1/np.delete(mfpt[i, :], i))
    efficiency_global = np.mean(efficiency_local)

    return efficiency_global


def get_ROInames(Network, path=None):
    '''
    Function to get a list of names of individual ROI regions in the given
    parcellation.
    '''

    if path is None:
        path = os.path.expanduser('~')

    f = (path + "/nnt-data/atl-cammoun2012/MNI152NLin2009aSym/" +
         "atl-Cammoun2012_space-MNI152NLin2009aSym_info.csv")

    scale = "scale"+Network['cammoun_id']

    df = pd.read_csv(f)
    df_scale = df.loc[df['scale'] == scale]

    ROInames = []
    for index, row in df_scale.iterrows():
        ROInames.append(row['label'])

    ROInames = np.array(ROInames)

    ROInames = ROInames[Network['node_mask']]

    return ROInames


def getPCAgene(genes, scaled=True, return_scores=False):
    '''
    Function to compute the Principal components of our
    [genes x brain regions] matrix (see Burt et al., 2018).

    Parameters
    -----------
    INPUTS:
        genes -> gene expression data, [ndarray; shape:(n_nodes, n_genes)]

    Returns
    -------
        PCs -> PCs of gene expression [ndarray; shape:(10, n_nodes)]
        ev  -> Explained variance ratio for PCs
    '''

    genes = genes.copy()

    if scaled:
        scaler = StandardScaler()
        genes = scaler.fit_transform(genes)

    pca = PCA(n_components=10)
    scores = pca.fit_transform(genes.T)
    ev = pca.explained_variance_ratio_
    PCs = pca.components_

    if not return_scores:
        return PCs, ev
    else:
        return PCs, ev, scores


def getPCs(A, n_components=10):
    "Function to get the principal components of a matrix"
    pca = PCA(n_components=n_components)
    pca.fit(A)
    ev = pca.explained_variance_ratio_
    PCs = pca.components_
    return PCs, ev


def _load_hemi_info(parcel, info_path):

    with open(info_path+"/hemi.pkl", "rb") as handle:
        hemi = pickle.load(handle)

    if parcel[0] == "s":
        hemi = hemi[parcel[1:]]
    else:
        n = parcel_to_n(parcel)
        hemi = hemi[n].reshape(-1)

    return hemi


def _load_subcortex_info(parcel, info_path):

    with open(info_path+"/subcortexNodes.pkl", "rb") as handle:
        subcortex = pickle.load(handle)

    if parcel[0] == 's':
        subcortex = subcortex[parcel[1:]]
    else:
        n = parcel_to_n(parcel)
        subcortex = subcortex[n]

    return subcortex
