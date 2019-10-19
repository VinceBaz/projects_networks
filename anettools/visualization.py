
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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