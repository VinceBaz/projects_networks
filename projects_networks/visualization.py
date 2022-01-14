import numpy as np
import matplotlib.pyplot as plt
import os
import numbers
from netneurotools.plotting import plot_fsaverage
from netneurotools.datasets import fetch_cammoun2012, fetch_schaefer2018
from . import colors
from mpl_toolkits.mplot3d import Axes3D  # noqa
import warnings


def plot_brain_surface(values, network, hemi=None, cmap="viridis", alpha=0.8,
                       colorbar=True, centered=False, vmin=None, vmax=None,
                       representation='surface'):
    '''
    Function to plot data on the brain, on a surface parcellation.

    PARAMETERS
    ----------
    values : ndarray (n,)
        Values to be plotted on the brain, where n is the number of nodes in
        the parcellation.
    network : dictionary
        Dictionary storing the network on associated with the values (to be
        used to identify the adequate surface parcellation)
    '''

    cortical_hemi_mask = network['hemi_mask'][network['subcortex_mask'] == 0]
    n = len(cortical_hemi_mask)

    if hemi is None:
        hemi = network['info']['hemi']

    if hemi == "L":
        scores = np.zeros((n))+np.mean(values)
        scores[cortical_hemi_mask == 1] = values
        values = scores
    elif hemi == "R":
        scores = np.zeros((n))+np.mean(values)
        scores[cortical_hemi_mask == 0] = values
        values = scores

    order = network["order"]
    noplot = network["noplot"]
    lh = network["lhannot"]
    rh = network["rhannot"]

    if os.path.isfile(lh) or os.path.isfile(rh) is False:
        fetch_cammoun2012(version='fsaverage')
        fetch_schaefer2018()

    # Adjust colormap based on parameters
    if centered is True:
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
                        data_kws={'representation': representation},
                        show_toolbar=True)

    return im


def plot_network(G, coords, edge_scores, node_scores, edge_cmap="Greys",
                 edge_alpha=0.25, edge_vmin=None, edge_vmax=None,
                 node_cmap="viridis", node_alpha=1, node_vmin=None,
                 nodes_color='black', node_vmax=None, linewidth=0.25, s=100,
                 projection=None, view="sagittal", view_edge=True,
                 ordered_node=False, axis=False, directed=False, figsize=None,
                 node_order=None):
    '''
    Function to draw (plot) a network of nodes and edges.

    Parameters
    ----------
    G : dict or (n,n) ndarray
        Dictionary storing general information about the network we wish to
        plot or an (n,n) ndarray storing the adjacency matrix of the network.
        Where 'n' is the number of nodes in the network.
    coords : (n, 3) ndarray
        Coordinates of the network's nodes.
    edge_scores: (n,n) ndarray
        ndarray storing edge scores for individual edges in the network. These
        scores will be used to color the edges.
    node_scores : (n,) ndarray
        ndarray storing node scores for individual nodes in the network. These
        scores will be used to color the nodes.
    node_vmin, node_vmax: float, default: None
        Minimal and maximal values of the nodes colors. If None, the min and
        max of the node_scores array will be used.

    Returns
    -------
    '''

    if isinstance(G, dict):
        G = G['adj']

    if not np.all(G == G.T) and not directed:
        warnings.warn(("network appears to be directed, yet 'directed' "
                       "parameter was set to 'False'. The values of the edges "
                       "may be wrong."))

    if figsize is None:
        figsize = (10, 10)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)

    # Identify all the edges in the network
    Edges = np.where(G > 0)

    # Get the color of the edges
    if edge_scores is None:
        edge_colors = np.full((len(Edges[0])), "black", dtype="<U10")
    else:
        edge_colors = colors.get_color_distribution(edge_scores[Edges],
                                                    cmap=edge_cmap,
                                                    vmin=edge_vmin,
                                                    vmax=edge_vmax)

    if node_scores is None:
        node_scores = nodes_color

    if projection is None:

        # Plot the edges
        if view_edge:
            for edge_i, edge_j, c in zip(Edges[0], Edges[1], edge_colors):

                x1 = coords[edge_i, 0]
                x2 = coords[edge_j, 0]
                y1 = coords[edge_i, 1]
                y2 = coords[edge_j, 1]

                if not directed:
                    ax.plot([x1, x2],
                            [y1, y2],
                            c=c,
                            linewidth=linewidth,
                            alpha=edge_alpha,
                            zorder=0)
                else:
                    len_x = (x2 - x1)
                    len_y = (y2 - y1)
                    ax.arrow(x1,
                             y1,
                             dx=len_x * 0.8,
                             dy=len_y * 0.8,
                             length_includes_head=True,
                             shape='left',
                             width=linewidth*0.1,
                             head_width=linewidth*0.15,
                             alpha=edge_alpha,
                             color=c,
                             zorder=0)

        # plot the nodes
        if not ordered_node:
            ax.scatter(coords[:, 0],
                       coords[:, 1],
                       c=node_scores,
                       cmap=node_cmap,
                       vmin=node_vmin,
                       vmax=node_vmax,
                       clip_on=False,
                       alpha=node_alpha,
                       s=s,
                       zorder=1)

        else:
            if node_order is None:
                order = np.argsort(node_scores)
            else:
                order = node_order

            # Order the colors and the sizes so that they match the scores
            if isinstance(s, numbers.Number):
                ordered_s = s
            else:
                ordered_s = s[order]
            if isinstance(node_scores, str):
                ordered_c = node_scores
                print("Warning!!! You cannot order scores that don't exist")
            else:
                ordered_c = node_scores[order]

            # Plot the nodes
            ax.scatter(coords[order, 0],
                       coords[order, 1],
                       c=ordered_c,
                       edgecolors='none',
                       cmap=node_cmap,
                       vmin=node_vmin,
                       vmax=node_vmax,
                       alpha=node_alpha,
                       s=ordered_s,
                       zorder=1)
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

        ax.scatter(coords[:, 0],
                   coords[:, 1],
                   coords[:, 2],
                   c=node_scores,
                   cmap=node_cmap,
                   # edgecolors='none',
                   edgecolors='face',
                   s=s,
                   zorder=1,
                   # depthshade=False,
                   depthshade=True,
                   vmin=node_vmin,
                   vmax=node_vmax)

        if view_edge:
            for edge_i, edge_j, c in zip(Edges[0], Edges[1], edge_colors):
                ax.plot([coords[edge_i, 0], coords[edge_j, 0]],
                        [coords[edge_i, 1], coords[edge_j, 1]],
                        [coords[edge_i, 2], coords[edge_j, 2]],
                        c=c,
                        alpha=edge_alpha,
                        linewidth=linewidth,
                        zorder=0)

        scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)

    if not axis:
        ax.axis('off')

    return fig, ax
