
import numpy as np
import matplotlib.pyplot as plt
from . import measures as m
import random


def plot_gradient(partition, coords, label=None, min_color=None,
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
        cbar = fig.colorbar(mapp[0], cax=cbar_ax, orientation='horizontal', pad=0.05)

        if label is not None:
            cbar.set_label(label, size=20)
            cbar.ax.tick_params(labelsize=20)

    if show is False:
        plt.close()


def assort_preserv_swap(A, M, g, verbose=False):

    assort = m.globalAssort(A, M)

    # Remember previous graph in the sequence of graphs
    prevG = A

    # Stores if swap was successful, and the edges that are swapped
    swapped = []
    swapped.append(True)
    edgeSwapped = []

    # while assort is less than goal assort
    while assort < g:

        edges = np.where(prevG == 1)

        # Choose 2 edges at random
        rand1 = random.randrange(0, len(edges[0]))
        rand2 = random.randrange(0, len(edges[0]))
        e1 = [edges[0][rand1], edges[1][rand1]]
        e2 = [edges[0][rand2], edges[1][rand2]]

        # if swap creates a self-loop or a multiedge, resample
        if e1[0] == e2[1] or e2[0] == e1[1]:
            swapped.append(False)
        elif prevG[e1[0], e2[1]] == 1 or prevG[e2[0], e1[1]] == 1:
            swapped.append(False)
        else:
            # swap edges
            newG = prevG.copy()
            newG[e1[0], e1[1]] = 0
            newG[e1[1], e1[0]] = 0
            newG[e2[0], e2[1]] = 0
            newG[e2[1], e2[0]] = 0
            newG[e1[0], e2[1]] = 1
            newG[e2[1], e1[0]] = 1
            newG[e2[0], e1[1]] = 1
            newG[e1[1], e2[0]] = 1

            # If new graph increases assortativity, keep swap
            assortNew = m.globalAssort(newG, M)
            if assortNew - assort > 0:
                prevG = newG
                swapped.append(True)
                edgeSwapped.append([e1, e2])
                assort = assortNew

                if verbose is True:
                    print(assort)

            else:
                swapped.append(False)

    return prevG, swapped, edgeSwapped
