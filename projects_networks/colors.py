import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap


def get_color_distribution(scores, cmap="viridis", vmin=None, vmax=None):

    '''
    Function to get a color for individual values of a distribution of scores.
    '''

    n = len(scores)

    if vmin is None:
        vmin = np.amin(scores)
    if vmax is None:
        vmax = np.amax(scores)

    cmap = cm.get_cmap(cmap, 256)
    new_colors = cmap(np.linspace(0, 1, 256))

    if vmin != vmax:
        scaled = (scores - vmin)/(vmax - vmin) * 255
    else:
        scaled = np.zeros((n)) + 128

    c = np.zeros((n, 4))
    for i in range(n):
        c[i] = new_colors[int(scaled[i]), :]

    return c


def get_cmap(colorList):
    '''
    Function to get a colormap from a list of colors
    '''
    n = len(colorList)
    c_all = np.zeros((256, 4))
    m = int(256/(n-1))
    for i in range(n):

        if isinstance(colorList[i], str):
            color = mpl.colors.to_rgba(colorList[i])
        else:
            color = colorList[i]

        if i == 0:
            c_all[:int(m/2)] = color
        elif i < n-1:
            c_all[((i-1)*m)+(int(m/2)):(i*m)+(int(m/2))] = color
        else:
            c_all[((i-1)*m)+(int(m/2)):] = color

    cmap = ListedColormap(c_all)

    return cmap
