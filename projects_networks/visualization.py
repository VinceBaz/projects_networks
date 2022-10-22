import numpy as np
import matplotlib.pyplot as plt
import os
import numbers
import cv2
from itertools import groupby
from operator import itemgetter
from netneurotools.plotting import plot_fsaverage
from netneurotools.datasets import fetch_cammoun2012, fetch_schaefer2018
from .colors import get_color_distribution
from .load_data import get_node_masks, get_general_parcellation_info
from mpl_toolkits.mplot3d import Axes3D  # noqa
import warnings
from neuromaps.parcellate import Parcellater
from neuromaps.images import annot_to_gifti, relabel_gifti
from neuromaps import transforms
from neuromaps.plotting import plot_surf_template


def plot_brain_surface(values, brain, hemi=None, cmap="viridis", alpha=0.8,
                       colorbar=True, centered=False, vmin=None, vmax=None,
                       representation='wireframe'):
    '''
    Function to plot data on the brain, on a surface parcellation.

    Parameters
    ----------
    values : ndarray (n,)
        Values to be plotted on the brain, where n is the number of nodes in
        the parcellation.
    brain : str or dictionary
        String describing the parcellation (e.g. {`68`, '114`, 's400`, etc.)
        to be used or dictionary storing the network data associated with the
        brainmap (to be used to identify the adequate surface parcellation).
    '''

    if hemi is None:
        hemi = brain['info']['hemi']

    if isinstance(brain, dict):
        cortical_hemi_mask = brain['hemi_mask'][brain['subcortex_mask'] == 0]
        order = brain["order"]
        noplot = brain["noplot"]
        lh = brain["lhannot"]
        rh = brain["rhannot"]
    else:
        _, hemi_mask, subcortex_mask = get_node_masks(None, N_hemi=hemi,
                                                      N_parcel=brain)
        cortical_hemi_mask = hemi_mask[subcortex_mask == 0]
        parcel_info = get_general_parcellation_info(brain)
        order = parcel_info[0]
        noplot = parcel_info[1]
        lh = parcel_info[2]
        rh = parcel_info[3]
    n = len(cortical_hemi_mask)

    if hemi == "L":
        scores = np.zeros((n))+np.mean(values)
        scores[cortical_hemi_mask == 'L'] = values
        values = scores
    elif hemi == "R":
        scores = np.zeros((n))+np.mean(values)
        scores[cortical_hemi_mask == 'R'] = values
        values = scores

    if os.path.isfile(lh) or os.path.isfile(rh) is False:
        fetch_cammoun2012(version='fsaverage')
        fetch_schaefer2018()

    # Adjust colormap based on parameters
    if centered:
        m = max(abs(np.amin(values)), np.amax(values))
        vmin = -m
        vmax = m
    else:
        if vmin is None:
            vmin = np.amin(values)
        if vmax is None:
            vmax = np.amax(values)

    # Plot the brain surface
    im = plot_fsaverage(values,
                        lhannot=lh,
                        rhannot=rh,
                        noplot=noplot,
                        order=order,
                        views=['lateral', 'm'],
                        vmin=vmin,
                        vmax=vmax,
                        colormap=cmap,
                        alpha=alpha,
                        colorbar=colorbar,
                        data_kws={'representation': representation,
                                  'line_width': 4.0},
                        show_toolbar=True,
                        )

    return im


def plot_surface(values, brain, hemi=None,  cmap="viridis", vmin=None,
                 vmax=None, centered=False, engine='matplotlib'):
    '''
    Function to plot data on a surface parcellation. The goal of this function
    is the same as `plot_brain_surface`, but relies on `neuromaps`' plotting
    function instead of `netneurotools`' plotting function.

    Parameters
    ----------
    values: (n,) array-like
        Values to be plotted on the brain, where n is the number of nodes in
        the parcellation.
    brain : str or dictionary
        String describing the parcellation (e.g. {`68`, '114`, 's400`, etc.)
        to be used or dictionary storing the network data associated with the
        brainmap (to be used to identify the adequate surface parcellation).
    '''

    if hemi is None:
        hemi = brain['info']['hemi']

    # Load information about the parcellation
    if isinstance(brain, dict):
        cortical_hemi_mask = brain['hemi_mask'][brain['subcortex_mask'] == 0]
        order = brain["order"]
        lh = brain["lhannot"]
        rh = brain["rhannot"]
    else:
        _, hemi_mask, subcortex_mask = get_node_masks(None, N_hemi=hemi,
                                                      N_parcel=brain)
        cortical_hemi_mask = hemi_mask[subcortex_mask == 0]
        parcel_info = get_general_parcellation_info(brain)
        order = parcel_info[0]
        lh = parcel_info[2]
        rh = parcel_info[3]
    n = len(cortical_hemi_mask)

    if hemi == "L":
        scores = np.zeros((n))+np.mean(values)
        scores[cortical_hemi_mask == 'L'] = values
        values = scores
    elif hemi == "R":
        scores = np.zeros((n))+np.mean(values)
        scores[cortical_hemi_mask == 'R'] = values
        values = scores

    if order == 'RL':
        values = np.concatenate((scores[cortical_hemi_mask == 'L'],
                                 scores[cortical_hemi_mask == 'R']),
                                axis=0)

    # Adjust colormap based on parameters
    if centered:
        m = max(abs(np.amin(values)), np.amax(values))
        vmin = -m
        vmax = m
    else:
        if vmin is None:
            vmin = np.amin(values)
        if vmax is None:
            vmax = np.amax(values)

    if engine == 'matplotlib':
        parcellation_maps = relabel_gifti(annot_to_gifti((lh, rh)))
        parcellater = Parcellater(parcellation_maps, 'fsaverage',
                                  resampling_target='parcellation')
        surface_values = parcellater.inverse_transform(values)
        density = transforms._estimate_density((surface_values,), hemi=None)[0]
        fig = plot_surf_template(
            surface_values, 'fsaverage', density, surf='pial', vmin=vmin,
            vmax=vmax, cmap=cmap, colorbar=True)

    return fig


def plot_network(A, coords, edge_scores, node_scores, edge_cmap="Greys",
                 edge_alpha=0.25, node_alpha=1, edge_vmin=None, edge_vmax=None,
                 nodes_color='black', node_kwargs=None,  edges_color='black',
                 linewidth=0.25, projection=None, view="sagittal",
                 view_edge=True, ordered_node=False, axis=False,
                 directed=False, figsize=None, node_order=None):
    '''
    Function to draw (plot) a network of nodes and edges.

    Parameters
    ----------
    A : (n, n) ndarray
        Array storing the adjacency matrix of the network. 'n' is the
        number of nodes in the network.
    coords : (n, 3) ndarray
        Coordinates of the network's nodes.
    edge_scores: (n, n) ndarray
        Array storing edge scores for individual edges in the network. These
        scores are used to color the edges.
    node_scores : (n) ndarray
        Array storing node scores for individual nodes in the network. These
        scores are used to color the nodes.
    edge_cmap: str
        Colormap from matplotlib.
    edge_alpha, node_alpha: float, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque)
    edge_vmin, edge_vmax: float, optional
        Minimal and maximal values of the edge colors. If None, the min and max
        of edge_scores  are used. Default: `None`
    node_kwargs: dict
        Dictionary of keyword arguments passed to `pyplot.scatter` used to
        personalize the visualization of the nodes in the network. Examples
        of commonly used kewyword arguments are: `cmap`, `edgecolors`, `size`.
    ordered_node: bool
        If True, nodes will be plotted in order specified by the node_order
        argument, or ordered according to the node scores.

    Returns
    -------
    fig: matplotlib.figure.Figure instance
        Figure instance of the drawn network.
    ax: matplotlib.axes.Axes instance
        Ax instance of the drawn network.
    '''

    if isinstance(A, dict):  # For compatibility
        warnings.warn(("In the future, the use of the network `dict` will be "
                      "deprecated"))
        A = A['adj']

    if not np.all(A == A.T) and not directed:
        warnings.warn(("The adjacency matrix is directed, but `directed` "
                       "parameter is set to `False`. The values of the edges "
                       "may be wrong."))

    if node_kwargs is None:
        node_kwargs = {}

    if 's' not in node_kwargs:
        node_kwargs['s'] = 100

    if figsize is None:
        figsize = (10, 10)

    fig, ax = plt.subplots(
        1, 1, figsize=figsize, subplot_kw={"projection": projection})

    # Identify all the edges in the network
    edges = np.where(A > 0)

    # Get the color of the edges
    if edge_scores is None:
        edge_colors = np.full((len(edges[0])), edges_color, dtype="<U10")
    else:
        edge_colors = get_color_distribution(
            edge_scores[edges], cmap=edge_cmap, vmin=edge_vmin, vmax=edge_vmax)

    # Get the color of the nodes
    if node_scores is None:
        node_scores = nodes_color
    node_colors = node_scores

    if projection is None:

        # order nodes (and adjust colors and sizes)
        if ordered_node:
            if node_order is None:
                node_order = np.argsort(node_scores, axis=None)
            if not isinstance(node_kwargs['s'], numbers.Number):
                node_kwargs['s'] = node_kwargs['s'][node_order]
            node_colors = node_colors[node_order]
        else:
            node_order = np.arange(len(A))

        # Plot the edges
        if view_edge:
            for edge_i, edge_j, c in zip(edges[0], edges[1], edge_colors):

                x1, x2 = coords[edge_i, 0], coords[edge_j, 0]
                y1, y2 = coords[edge_i, 1], coords[edge_j, 1]

                if not directed:
                    ax.plot(
                        [x1, x2], [y1, y2], c=c, linewidth=linewidth,
                        alpha=edge_alpha, zorder=0)
                else:
                    len_x = (x2 - x1)
                    len_y = (y2 - y1)
                    ax.arrow(
                        x1, y1,  dx=len_x * 0.8, dy=len_y * 0.8,
                        length_includes_head=True, shape='left',
                        width=linewidth*0.1, head_width=linewidth*0.15,
                        alpha=edge_alpha, color=c, zorder=0)

        # plot the nodes
        ax.scatter(
            coords[node_order, 0], coords[node_order, 1], c=node_colors,
            alpha=node_alpha, zorder=1, **node_kwargs)
        ax.set_aspect('equal')

    elif projection == "3d":

        if isinstance(view, str):
            # axial view of the brain
            if view == "axial":
                ax.view_init(90, 0)

            # sagittal view of the brain
            elif view == "sagittal":
                ax.view_init(0, 0)
        else:
            ax.view_init(view[0], view[1])

        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2], c=node_scores,
            zorder=1, depthshade=True, **node_kwargs)

        if view_edge:
            for edge_i, edge_j, c in zip(edges[0], edges[1], edge_colors):
                ax.plot([coords[edge_i, 0], coords[edge_j, 0]],
                        [coords[edge_i, 1], coords[edge_j, 1]],
                        [coords[edge_i, 2], coords[edge_j, 2]],
                        c=c, alpha=edge_alpha, linewidth=linewidth, zorder=0)

        scaling = np.array(
            [getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']
            )
        ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

    if not axis:
        ax.axis('off')

    return fig, ax


def save_cropped_background(image_file):
    '''
    Function to crop the background from a saved `.png` surface images and
    save the images obtained into a `.svg` file
    '''

    full_img = cv2.imread(image_file)

    mask_full = ((full_img[:, :, 0] == 255) &
                 (full_img[:, :, 1] == 255) &
                 (full_img[:, :, 2] == 255))

    rows_groups = []
    empty_rows_id = np.where(np.all(mask_full == 1, axis=1) == 0)[0]
    for k, g in groupby(enumerate(empty_rows_id),
                        lambda i_x: i_x[0] - i_x[1]):
        rows_groups.append(list(map(itemgetter(1), g)))

    img_top = full_img[rows_groups[0], :, :]
    img_bot = full_img[rows_groups[3], :, :]

    img_counter = 0
    for img in [img_top, img_bot]:
        mask = ((img[:, :, 0] == 255) &
                (img[:, :, 1] == 255) &
                (img[:, :, 2] == 255))

        columns_groups = []
        empty_columns_id = np.where(np.all(mask == 1, axis=0) == 0)[0]
        for k, g in groupby(enumerate(empty_columns_id),
                            lambda i_x: i_x[0] - i_x[1]):
            columns_groups.append(list(map(itemgetter(1), g)))

        for group in columns_groups:
            cropped_img = img[:, group, :]
            cv2.imwrite(f"{image_file[:-4]}_cropped_{img_counter}.png",
                        cropped_img)
            img_counter += 1
