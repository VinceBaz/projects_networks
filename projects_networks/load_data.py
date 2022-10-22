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
import warnings
import pandas as pd
from . import null


def load_network(kind, parcel, data="lau", weights='log', subset="all",
                 version=1, hemi="both", path=None):
    '''
    Function to load a dictionary containing information about the specified
    brain network.

    Parameters
    ----------
    kind : string
        Either 'SC' or 'FC'.
    hemi : string
        Either "both", "L" or "R".
    weights " string
        The weights of the edges. The options  "normal", "log" or "binary".
        The default is "log".
    data : string
        Either "HCP" or "lau".
    parcel : string
        Either "68", "114", ... [if 'lau'] / "s400", "s800" [if "HCP"]
    version : int
        Version of the network.
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
    Network["info"]["weights"] = weights
    Network["info"]["version"] = version
    Network["info"]["subset"] = subset

    # Modify parameter names to what they are in file names
    version = '' if version == 1 else '_v' + str(version)
    subset = '' if subset == 'all' else subset
    hemi = '' if hemi == 'both' else hemi

    # Store important paths for loading the relevant data
    main_path = f'{path}/brainNetworks/{data}/'
    network_path = (f'{main_path}/matrices/consensus/{subset}{kind}{parcel}'
                    f'{hemi}{version}/')
    matrix_path = f'{network_path}/{weights}'

    # Store general information about the network's parcellation
    parcel_info = get_general_parcellation_info(parcel)
    Network['order'] = parcel_info[0]
    Network['noplot'] = parcel_info[1]
    Network['lhannot'] = parcel_info[2]
    Network['rhannot'] = parcel_info[3]
    Network['atlas'] = parcel_info[4]
    Network['parcellation_name'] = parcel_info[5]

    # Load the cammoun_id of the parcellation, if Cammoun (i.e. 033, 060, etc.)
    if parcel[0] != 's':
        Network['cammoun_id'] = parcel_to_n(parcel)

    # masks
    masks = get_node_masks(Network)
    Network['node_mask'] = masks[0]
    Network['hemi_mask'] = masks[1]
    Network['subcortex_mask'] = masks[2]

    # hemisphere
    Network['hemi'] = get_hemi(Network)

    # coordinates
    Network['coords'] = get_coordinates(Network, path=main_path)

    # Adjacency matrix
    Network['adj'], last_modified = get_adjacency(Network, matrix_path,
                                                  minimal_processing=True,
                                                  return_last_modified=True)

    # Test whether the network is connected. Raise a warning if not...
    if not np.all(bct.reachdist(Network['adj'])[0]):
        warnings.warn(("This brain network appears to be disconnected. This "
                       "might cause problem for the computation of the other "
                       "measures"))

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
    Network['sp'] = get_shortest_path(Network, matrix_path=matrix_path,
                                      last_modified=last_modified)

    # diffusion embedding
    de = compute_diffusion_map(Network['adj'],
                               n_components=10,
                               return_result=True,
                               skip_checks=True)
    Network["de"] = de[0]
    Network["de_extra"] = de[1]

    # Principal components
    Network['PCs'], Network['PCs_ev'] = getPCs(Network['adj'])

    # eigenvector centrality
    Network["ec"] = bct.eigenvector_centrality_und(Network['adj'])

    # mean first passage time
    Network["mfpt"] = bct.mean_first_passage_time(Network['adj'])

    # betweenness centrality
    Network['bc'] = get_betweenness(Network, matrix_path=matrix_path,
                                    last_modified=last_modified)

    # routing efficiency
    Network["r_eff"] = efficiency(Network)

    # diffusion efficiency
    Network["d_eff"] = efficiency_diffusion(Network)

    # subgraph centrality
    Network["subc"] = bct.subgraph_centrality(Network["adj"])

    # closeness centrality
    Network['clo'] = 1/np.mean(Network['sp'], axis=0)

    # communities + participation coefficient
    path = matrix_path + "/communities/"
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

    # Edge lengths
    if (data == "HCP") and (kind == "SC"):
        path = main_path+"matrices/"+subset+kind+parcel+hemi+"_lengths.npy"
        if os.path.exists(path):
            Network["lengths"] = np.load(path)

    # streamline connection lengths
    path = network_path + "/len.npy"
    if os.path.exists(path):
        Network['len'] = np.load(path)

    # ROI names
    if parcel[0] != "s":
        Network['ROInames'] = get_ROI_names(Network)

    # geodesic distances between nodes
    if parcel[0] == "s":
        n = parcel[1:]
        fname_l = n + "Parcels7Networks_lh_dist.csv"
        fname_r = n + "Parcels7Networks_rh_dist.csv"
    else:
        fname_l = "scale" + Network['cammoun_id'] + "_lh_dist.csv"
        fname_r = "scale" + Network['cammoun_id'] + "_rh_dist.csv"
    Network['geo_dist_L'] = pd.read_csv(main_path+"/geodesic/medial/" + fname_l,
                                        header=None).values
    Network['geo_dist_R'] = pd.read_csv(main_path+"/geodesic/medial/" + fname_r,
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

    node_mask, _, sub = get_node_masks(None, N_hemi=hemi, N_parcel=parcel)

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
        order, _, lhannot, rhannot, _ = get_general_parcellation_info(parcel)
        ANN['spin'] = null.generate_spins(parcel, lhannot, rhannot,
                                          order, info_path, hemi=hemi)
        np.save(path, ANN['spin'])

    # Morphometric property (T1w/T2w) | HCP Reinder
    path = mainPath+"annotations/T1wT2w/"+subset+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["t1t2"] = np.mean(np.load(path), axis=0)

    # Morphometric property (Thickness) | HCP Reinder
    path = mainPath+"annotations/thi/"+subset+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN["thi"] = np.mean(np.load(path), axis=0)

    if data == 'lau':
        # T1w/T2w (from neuromaps)
        path = f"{mainPath}annotations/T1wT2w/{parcel_to_n(parcel)}.npy"
        if os.path.exists(path):
            ANN["t1t2"] = np.load(path)[np.delete(node_mask, sub)]
        # Thickness (from neuromaps)
        path = f"{mainPath}annotations/thi/{parcel_to_n(parcel)}.npy"
        if os.path.exists(path):
            ANN["thi"] = np.load(path)[np.delete(node_mask, sub)]

    # Receptor Excitatory/Inhibitory ratio
    path = f"../data/receptor/EI_ratio/{parcel}{hemi}_scaled_no_bg.npy"
    if os.path.exists(path):
        ANN['EI_ratio'] = np.load(path)

    # Receptor density
    path = f'../data/receptor/mean_density/density_{parcel}{hemi}_scaled_no_bg.npy'
    if os.path.exists(path):
        ANN['receptor_den'] = np.load(path)

    # Receptor Principal Components
    path = '../data/receptor/PCs/PCs_'+parcel+hemi+".npy"
    if os.path.exists(path):
        ANN['receptor_PCs'] = np.load(path)

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

    path = matrix_path + "/adj.npy"

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
    Function to get some general information about the parcellation.

    Parameters
    ----------
    parcel: string
        Short name for the specific parcellation. For instance: "68", "114",
        "s400", etc.
    '''

    home = os.path.expanduser("~")

    # Get the general information, for the Schaefer parcellations
    if parcel.startswith('s'):
        n = parcel[1:]
        order = "LR"
        noplot = [b'Background+FreeSurfer_Defined_Medial_Wall',
                  b'']
        path2files = (home +
                      '/'
                      'nnt-data/'
                      'atl-schaefer2018/')
        lhannot = (path2files +
                   "fsaverage/"
                   "atl-Schaefer2018_space-fsaverage_"
                   "hemi-L_desc-" + n + "Parcels7Networks_"
                   "deterministic.annot")
        rhannot = (path2files +
                   "fsaverage/"
                   "atl-Schaefer2018_space-fsaverage_"
                   "hemi-R_desc-" + n + "Parcels7Networks_"
                   "deterministic.annot")
        atlas = (path2files +
                 'MNI152/'
                 'Schaefer2018_' + n + 'Parcels_7Networks_order_'
                 'FSLMNI152_1mm.nii.gz')
        name = 'schaefer'

    # Get the general information for the Cammoun parcellations
    else:
        n = parcel_to_n(parcel)
        order = "RL"
        noplot = None
        path2files = (home +
                      "/"
                      "nnt-data/"
                      'atl-cammoun2012/')
        lhannot = (path2files +
                   'fsaverage/'
                   'atl-Cammoun2012_space-fsaverage_'
                   'res-' + n + "_hemi-L_deterministic.annot")
        rhannot = (path2files +
                   'fsaverage/'
                   'atl-Cammoun2012_space-fsaverage_'
                   'res-' + n + "_hemi-R_deterministic.annot")
        atlas = (path2files +
                 'MNI152NLin2009aSym/'
                 'atl-Cammoun2012_space-MNI152NLin2009aSym_res-' +
                 n + '_deterministic.nii.gz')
        name = 'cammoun'

    return order, noplot, lhannot, rhannot, atlas, name


def get_node_masks(N, N_hemi=None, N_parcel=None):
    '''
    Function to get a mask of the nodes of this particular network (1), given
    the original parcellation (0).

    Parameters
    ----------
    N : dict
        Dictionary storing relevant information about the network of interest.
    N_hemi: {'L', 'R', 'both'}
        General hemisphere information associated with the network.
        Default: {None}
    N_parcel: str
        General information about the parcellation associated with the network.
        For instance: {'68', '114', '219', etc.}
    '''

    # Load info about the network
    if N_hemi is None:
        N_hemi = N['info']['hemi']
    if N_parcel is None:
        N_parcel = N['info']['parcel']

    # Load general info about hemispheres and subcortex
    hemi_mask = _load_hemi_info(N_parcel)
    subcortex_nodes = _load_subcortex_info(N_parcel)

    # Get subcortex mask
    n_nodes = len(hemi_mask)
    subcortex_mask = np.zeros((n_nodes), dtype=bool)
    subcortex_mask[subcortex_nodes] = True

    # Figure out whether this parcellation contains the subcortex
    subcortex = False
    if N_parcel in ['83', '129', '234', '463', '1015']:
        subcortex = True

    # Create a node mask for the network
    if not subcortex:
        if N_hemi == 'L':
            node_mask = np.logical_and(hemi_mask == 'L', subcortex_mask == 0)
        elif N_hemi == 'R':
            node_mask = np.logical_and(hemi_mask == 'R', subcortex_mask == 0)
        elif N_hemi == 'both':
            node_mask = (subcortex_mask == 0)
    elif subcortex:
        if N_hemi == 'L':
            node_mask = (hemi_mask == 'L')
        elif N_hemi == 'R':
            node_mask = (hemi_mask == 'R')
        elif N_hemi == 'both':
            node_mask = np.ones((n_nodes), dtype=bool)

    return node_mask, hemi_mask, subcortex_mask


def get_hemi(Network, node_mask=None, network_parcel=None):
    '''
    Function to get hemisphere information for the nodes in the network.
    This hemispheric information is viewed as an annotation. In other words
    the information is only given for the nodes of the network (so we rely on
    our network's 'node_mask' to extract the nodes of interest)
    '''
    # Get node mask of network
    if node_mask is None:
        mask = Network['node_mask']

    # Get some information about the network of interest (HCP or lau)
    if network_parcel is None:
        network_parcel = Network['info']['parcel']

    # Get information about hemispheres for the given parcellation
    hemi = _load_hemi_info(network_parcel)

    return hemi[mask]


def get_shortest_path(Network, matrix_path=None, last_modified=0):
    '''
    Function to get the shortest path of a network. If a matrix_path is given,
    this function will try to load the shortest path from this matrix path,
    if the shortest paths have been generated after the adjacency matrix itself
    has been generetaed.
    '''

    if matrix_path is not None:

        path = matrix_path + "/sp.npy"

        if not os.path.exists(path):

            print("shortest path not found")
            print("computing shortest path...")

            sp, _ = bct.distance_wei(Network['inv_adj'])
            np.save(path, sp)

        elif os.path.getmtime(path) < last_modified:

            print("new adjacency matrix was found")
            print("computing shortest paths...")

            sp, _ = bct.distance_wei(Network['inv_adj'])
            np.save(path, sp)

        else:

            sp = np.load(path)

    else:
        sp, _ = bct.distance_wei(Network['inv_adj'])

    return sp


def get_betweenness(Network, matrix_path, last_modified=0):
    '''
    Function to get the betweenness centrality of of a network's nodes. If a
    matrix_path is given, this function will try to load the centrality scores
    using this matrix path. If these scores have been generated before the
    adjacency matrix itself has been generated, then the scores will be
    recomputed.
    '''

    if matrix_path is not None:

        path = matrix_path + "/bc.npy"

        if not os.path.exists(path):

            print("betweenness centrality not found")
            print("computing betweenness centrality...")

            bc = bct.betweenness_wei(Network["inv_adj"])
            np.save(path, bc)

        elif os.path.getmtime(path) < last_modified:

            print("new adjacency matrix was found")
            print("computing betweenness centrality...")

            bc = bct.betweenness_wei(Network["inv_adj"])
            np.save(path, bc)

        else:

            bc = np.load(path)

    else:
        bc = bct.betweenness_wei(Network["inv_adj"])

    return bc


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


def get_ROI_names(Network, path=None):
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


def getPCAgene(genes, scaled=True, return_scores=False, n_components=10):
    '''
    Function to compute the Principal components of our
    [genes x brain regions] matrix (see Burt et al., 2018).

    Parameters
    -----------
    genes: (n_nodes, n_genes) ndarray
        Gene expression data

    Returns
    -------
        PCs -> PCs of gene expression [ndarray; shape:(10, n_nodes)]
        ev  -> Explained variance ratio for PCs
    '''

    genes = genes.copy()

    if scaled:
        scaler = StandardScaler()
        genes = scaler.fit_transform(genes)

    pca = PCA(n_components=n_components)
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


def _load_hemi_info(parcel):

    if parcel.startswith('s'):
        parc_name = 'Schaefer2018'
        scale = parcel[1:]
    else:
        parc_name = 'Cammoun2012'
        try:
            scale = parcel_to_n(parcel)
        except KeyError:
            scale = parcel

    file_dir = os.path.dirname(__file__)
    csv = pd.read_csv((f"{file_dir}/data/parcellation_info/"
                       f"{parc_name}_{scale}.csv"))
    hemi = csv['hemisphere'].to_numpy(dtype='str')

    return hemi


def _load_subcortex_info(parcel):

    if parcel.startswith('s'):
        parc_name = 'Schaefer2018'
        scale = parcel[1:]
    else:
        parc_name = 'Cammoun2012'
        try:
            scale = parcel_to_n(parcel)
        except KeyError:
            scale = parcel

    file_dir = os.path.dirname(__file__)
    csv = pd.read_csv((f"{file_dir}/data/parcellation_info/"
                       f"{parc_name}_{scale}.csv"))
    structure = csv['structure'].to_numpy(dtype='str')

    subcortex = np.where(structure == 'subcortex')[0].tolist()

    return subcortex
