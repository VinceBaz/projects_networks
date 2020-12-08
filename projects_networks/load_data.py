# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:12:05 2020

Code to load network-related data from a "data" folder stored locally
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

    # Spin tests
    path = mainPath+"surrogates/spins_"+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["spin"] = np.load(path)

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

    mainPath = path+"/brainNetworks/"+data+"/"
    home = os.path.expanduser("~")

    Network = {}

    Network["info"] = {}
    Network["info"]["kind"] = kind
    Network["info"]["parcel"] = parcel
    Network["info"]["data"] = data
    Network["info"]["hemi"] = hemi
    Network["info"]["binary"] = binary
    Network["info"]["version"] = version
    Network["info"]["subset"] = subset

    if version == 1:
        version = ''
    else:
        version = "_v2"

    if binary is True:
        binary = "b"
    else:
        binary = ''

    if subset == "all":
        subset = ''

    if hemi == "both":
        hemi = ''

    matrixPath = mainPath+"matrices/"+subset+kind+parcel+hemi+binary+version

    # hemisphere
    Network["hemi"] = np.load(matrixPath+"/hemi.npy")

    # Adjacency matrix
    path = matrixPath+".npy"
    A = np.load(path)

    # Look at time when file was last modified
    last_modified = os.path.getmtime(path)

    # set negative values to 0, fill diagonal, make symmetric
    A[A < 0] = 0
    np.fill_diagonal(A, 0)
    A = (A + A.T)/2
    Network["adj"] = A

    # coordinates
    path = mainPath+"coords/coords"+parcel+hemi+".npy"
    Network["coords"] = np.load(path)

    # node strength
    Network["str"] = np.sum(A, axis=0)

    # Inverse of adjacency matrix
    inv = A.copy()
    inv[A > 0] = 1/inv[A > 0]
    Network["inv_adj"] = inv

    # distance
    Network["dist"] = cdist(Network["coords"], Network["coords"])

    # clustering coefficient
    Network["cc"] = bct.clustering_coef_wu(A)

    # shortest path
    #
    # Loaded from saved file...
    # IF file not found OR Adjacency was modified after creation,
    # then recompute measure
    path = matrixPath+"/sp.npy"

    if os.path.exists(path) is False:

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
    de = compute_diffusion_map(A, n_components=10, return_result=True)
    Network["de"] = de[0]
    Network["de_extra"] = de[1]

    # Principal components
    Network['PCs'], Network['PCs_ev'] = getPCs(Network['adj'])

    # eigenvector centrality
    Network["ec"] = bct.eigenvector_centrality_und(A)

    # mean first passage time
    Network["mfpt"] = bct.mean_first_passage_time(A)

    # betweenness centrality
    #
    # Loaded from saved file...
    # IF file not found OR Adjacency was modified after creation,
    # then recompute measure
    path = matrixPath+"/bc.npy"
    if os.path.exists(path) is False:

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
                ppc = bct.participation_coef(A, Network["ci"][i])
                Network["ppc"].append(ppc)

    if (data == "HCP") and (kind == "SC"):
        path = mainPath+"matrices/"+subset+kind+parcel+hemi+"_lengths.npy"
        if os.path.exists(path) is True:
            Network["lengths"] = np.load(path)

    # streamline connection lengths
    path = matrixPath+"/len.npy"
    if os.path.exists(path) is True:
        Network['len'] = np.load(path)

    # network information
    if parcel[0] == "s":
        n = parcel[1:]
        Network["order"] = "LR"
        Network["noplot"] = [b'Background+FreeSurfer_Defined_Medial_Wall',
                             b'']
        Network["lhannot"] = (home+"/"
                              "nnt-data/"
                              "atl-schaefer2018/"
                              "fsaverage/"
                              "atl-Schaefer2018_space-fsaverage_"
                              "hemi-L_desc-"+n+"Parcels7Networks_"
                              "deterministic.annot")
        Network["rhannot"] = (home+"/"
                              "nnt-data/"
                              "atl-schaefer2018/"
                              "fsaverage/"
                              "atl-Schaefer2018_space-fsaverage_"
                              "hemi-R_desc-"+n+"Parcels7Networks_"
                              "deterministic.annot")
    else:
        n = parcel_to_n(parcel)
        Network["order"] = "RL"
        Network["noplot"] = None
        Network["lhannot"] = (home+"/"
                              "nnt-data/"
                              "atl-cammoun2012/"
                              "fsaverage/"
                              "atl-Cammoun2012_space-fsaverage_"
                              "res-"+n+"_hemi-L_deterministic.annot")
        Network["rhannot"] = (home+"/"
                              "nnt-data/"
                              "atl-cammoun2012/"
                              "fsaverage/"
                              "atl-Cammoun2012_space-fsaverage_"
                              "res-"+n+"_hemi-R_deterministic.annot")
        Network['cammoun_id'] = n

    # Node mask
    Network['node_mask'] = get_node_mask(Network, path=mainPath)

    # ROI names
    if parcel[0] != "s":
        Network['ROInames'] = get_ROInames(Network)

    return Network


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


def get_node_mask(N, path="../data/brainNetworks/lau"):
    '''
    Function to get a mask of the nodes of this particular network (1), given
    the original parcellation (0).
    '''
    network_hemi = N['info']['hemi']

    if N['info']['data'] == 'lau':

        info_path = path+"/matrices/general_info"

        # Load info about which nodes are in which hemisphere
        with open(info_path+"/hemi.pkl", "rb") as handle:
            hemi = pickle.load(handle)
        hemi = hemi[N['cammoun_id']].reshape(-1)

        # Load info about which nodes are in the subcortex
        with open(info_path+"/subcortexNodes.pkl", "rb") as handle:
            subcortexNodes = pickle.load(handle)
        subcortexNodes = subcortexNodes[N['cammoun_id']]

        n = len(hemi)
        mask = np.zeros((n), dtype=bool)

        # Figure out whether this parcellation contains the subcortex
        subcortex = False
        if N["info"]["parcel"] in ['83', '129', '234', '463', '1015']:
            subcortex = True

        node_type = np.zeros(n)
        node_type[subcortexNodes] = 1

        if subcortex is False:
            if network_hemi == 'L':
                network_nodes = np.where((hemi == 1) & (node_type == 0))[0]
            elif network_hemi == 'R':
                network_nodes = np.where((hemi == 0) & (node_type == 0))[0]
            else:
                network_nodes = np.where((node_type == 0))[0]
        elif subcortex is True:
            if network_hemi == 'L':
                network_nodes = np.where(hemi == 1)[0]
            elif network_hemi == 'R':
                network_nodes = np.where(hemi == 0)[0]
            else:
                network_nodes = np.ones((n), dtype=bool)

    elif N['info']['data'] == 'HCP':

        n = len(N['adj'])
        mask = np.zeros((n), dtype=bool)
        hemi = N['hemi']

        if network_hemi == 'L':
            network_nodes = np.where((hemi == 1))[0]
        elif network_hemi == 'R':
            network_nodes = np.where((hemi == 0))[0]
        else:
            network_nodes = np.arange(n)

    mask[network_nodes] = True

    return mask


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

    if return_scores is False:
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
