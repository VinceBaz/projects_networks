import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
import os
from netneurotools.plotting import plot_fsaverage as p_fsa
from netneurotools.datasets import fetch_cammoun2012, fetch_schaefer2018


def plot_brain_surface(values, parcel="s", n="400", hemi="L", cmap="viridis",
                       colorbar=True, center=None, download=True):
    '''
    Function to plot data on the brain, on a surface parcellation.
    ------
    INPUTS
    ------
    -> values [ndarray (n,)] : Values to be plotted on the brain, where n is
    the number of nodes in the parcellation.
    -> download [Bool] :  Boolean on whether the choosen parcellation is to
    be downloaded if the files are not found in the home directory
    '''

    home = expanduser("~")

    if (hemi == "L") & (parcel == "lau"):
        if n == "500":
            scores = np.zeros((1000))+np.mean(values)
            scores[501:] = values
            values = scores

    if (hemi == "L") & (parcel == "s"):
        if n == "400":
            scores = np.zeros((400))+np.mean(values)
            scores[:200] = values
            values = scores
        if n == "800":
            scores = np.zeros((800))+np.mean(values)
            scores[:400] = values
            values = scores

    if parcel == "lau":
        order = "RL"
        noplot = None

        lh = (home+"/"
              "nnt-data/"
              "atl-cammoun2012/"
              "fsaverage/"
              "atl-Cammoun2012_space-fsaverage_"
              "res-"+n+"_hemi-L_deterministic.annot")
        rh = (home+"/"
              "nnt-data/"
              "atl-cammoun2012/"
              "fsaverage/"
              "atl-Cammoun2012_space-fsaverage_"
              "res-"+n+"_hemi-R_deterministic.annot")
        if os.path.isfile(lh) or os.path.isfile(rh) is False:
            fetch_cammoun2012(version='fsaverage')

    else:
        order = "LR"
        noplot = [b'Background+FreeSurfer_Defined_Medial_Wall', b'']
        lh = (home+"/"
              "nnt-data/"
              "atl-schaefer2018/"
              "fsaverage/"
              "atl-Schaefer2018_space-fsaverage_"
              "hemi-L_desc-"+n+"Parcels7Networks_deterministic.annot")
        rh = (home+"/"
              "nnt-data/"
              "atl-schaefer2018/"
              "fsaverage/"
              "atl-Schaefer2018_space-fsaverage_"
              "hemi-R_desc-"+n+"Parcels7Networks_deterministic.annot")
        if os.path.isfile(lh) or os.path.isfile(rh) is False:
            fetch_schaefer2018()

    im = p_fsa(values,
               lhannot=lh,
               rhannot=rh,
               noplot=noplot,
               order=order,
               views=['lateral', 'm'],
               vmin=np.amin(values),
               vmax=np.amax(values),
               colormap=cmap,
               colorbar=colorbar,
               center=center)

    return im


def plot_brain_dot(partition, coords, label=None, min_color=None,
                   max_color=None, colormap="viridis", colorbar=True, size=500,
                   show=True, dpi=100, norm=None):

    if min_color is None:
        min_color = np.amin(partition)
    if max_color is None:
        max_color = np.amax(partition)

    fig = plt.figure(figsize=(35, 10), dpi=dpi)

    orienX = [0, 0, 90]
    orienY = [270, 180, 180]
    ax = [None, None, None]
    mapp = [None, None, None]

    for k in range(3):

        ax[k] = fig.add_subplot(1, 3, k+1, projection='3d')
        ax[k].grid(True)
        ax[k].axis('off')

        mapp[k] = ax[k].scatter(xs=coords[:, 0],
                                ys=coords[:, 1],
                                zs=coords[:, 2],
                                c=partition,
                                vmin=min_color,
                                vmax=max_color,
                                s=size,
                                cmap=colormap,
                                edgecolors='k',
                                norm=norm)

        ax[k].view_init(orienX[k], orienY[k])
        ax[k].set(xlim=0.6 * np.array(ax[k].get_xlim()),
                  ylim=0.6 * np.array(ax[k].get_ylim()),
                  zlim=0.6 * np.array(ax[k].get_zlim()))

    plt.subplots_adjust(wspace=0, left=0, right=1, bottom=0.1, hspace=0)

    if colorbar is True:

        cbar_ax = fig.add_axes([0.3, 0, 0.4, 0.10])
        cbar = fig.colorbar(mapp[0],
                            cax=cbar_ax,
                            orientation='horizontal',
                            pad=0.05)

        if label is not None:
            cbar.set_label(label, size=20)
            cbar.ax.tick_params(labelsize=20)

    if show is False:
        plt.close()
