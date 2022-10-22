import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgba

from palettable.colorbrewer.diverging import (
    Spectral_4_r, Spectral_11_r, Spectral_4, Spectral_7, Spectral_11,
    RdBu_11_r, RdBu_9_r, PuOr_11, PiYG_11)
from palettable.colorbrewer.sequential import (
    Reds_3, Reds_9, Oranges_9, Blues_3, Blues_9, Greens_9, Purples_9, GnBu_9,
    Greys_9_r, Greys_7, Greys_9)
from palettable.cartocolors.sequential import (
    SunsetDark_7, agSunset_7_r, agSunset_7, PurpOr_7, Peach_7)
from palettable.cmocean.sequential import Ice_20
from palettable.cmocean.diverging import Delta_20, Curl_20
from palettable.colorbrewer.qualitative import (
    Pastel1_4, Pastel1_5, Pastel1_6, Set1_4, Set1_5, Set1_6, Set1_8, Set3_10)
from palettable.cartocolors.qualitative import Bold_4
from palettable.lightbartlein.diverging import RedYellowBlue_11_r


def get_color_distribution(scores, cmap="viridis", vmin=None, vmax=None,
                           default_color='black'):
    '''
    Function to get a color for individual values of a distribution of scores.
    '''

    if scores.min() == scores.max():
        c = np.full((len(scores)), default_color, dtype="<U10")
    else:
        c = cm.get_cmap(cmap)(mpl.colors.Normalize(vmin, vmax)(scores))

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


def cmap_ignore_negative(cmap, ignored_color='lightgray'):
    '''
    Function to get a colormap with the negative scores being a different
    color that is not part of the colormap (e.g. lightgray).
    '''
    newcolors = np.zeros((256, 4))
    spectral_4 = cm.get_cmap(cmap, 128)
    newcolors[128:, :] = spectral_4(np.linspace(0, 1, 128))
    lightgray = np.array(to_rgba(ignored_color))
    newcolors[:128, :] = lightgray
    newcmp = ListedColormap(newcolors)

    return newcmp


def get_colormaps():

    cmaps = {}

    # Colorbrewer | Diverging
    cmaps['Spectral_4_r'] = Spectral_4_r.mpl_colormap
    cmaps['Spectral_11_r'] = Spectral_11_r.mpl_colormap
    cmaps['Spectral_4'] = Spectral_4.mpl_colormap
    cmaps['Spectral_7'] = Spectral_7.mpl_colormap
    cmaps['Spectral_11'] = Spectral_11.mpl_colormap
    cmaps['RdBu_11_r'] = RdBu_11_r.mpl_colormap
    cmaps['RdBu_9_r'] = RdBu_9_r.mpl_colormap
    cmaps['PuOr_11'] = PuOr_11.mpl_colormap
    cmaps['PiYG_11'] = PiYG_11.mpl_colormap

    # Colorbrewer | Sequential
    cmaps['Reds_3'] = Reds_3.mpl_colormap
    cmaps['Reds_9'] = Reds_9.mpl_colormap
    cmaps['Blues_3'] = Blues_3.mpl_colormap
    cmaps['Blues_9'] = Blues_9.mpl_colormap
    cmaps['Purples_9'] = Purples_9.mpl_colormap
    cmaps['Greens_9'] = Greens_9.mpl_colormap
    cmaps['Oranges_9'] = Oranges_9.mpl_colormap
    cmaps['GnBu_9'] = GnBu_9.mpl_colormap
    cmaps['Greys_7'] = Greys_7.mpl_colormap
    cmaps['Greys_9'] = Greys_9.mpl_colormap
    cmaps['Greys_9_r'] = Greys_9_r.mpl_colormap

    # Colorbrewer | Qualitative
    cmaps['Set1_4'] = Set1_4.mpl_colormap
    cmaps['Set1_8'] = Set1_8.mpl_colormap
    cmaps['Set3_10'] = Set3_10.mpl_colormap

    # Cartocolors | Sequential
    cmaps['agSunset_7_r'] = agSunset_7_r.mpl_colormap
    cmaps['agSunset_7'] = agSunset_7.mpl_colormap
    cmaps['PurpOr_7'] = PurpOr_7.mpl_colormap
    cmaps['SunsetDark_7'] = SunsetDark_7.mpl_colormap
    cmaps['Peach_7'] = Peach_7.mpl_colormap

    # Cmocean | Sequential
    cmaps['Ice_20'] = Ice_20.mpl_colormap

    # Cmocean | Diverging
    cmaps['Delta_20'] = Delta_20.mpl_colormap
    cmaps['Curl_20'] = Curl_20.mpl_colormap

    # LightBartlein | Diverging
    cmaps['RedYellowBlue_11_r'] = RedYellowBlue_11_r.mpl_colormap

    return cmaps


def get_hexcolors():

    hex_colors = {}

    # Colorbrewer | Qualitative
    hex_colors['Pastel1_4'] = Pastel1_4.hex_colors
    hex_colors['Pastel1_5'] = Pastel1_5.hex_colors
    hex_colors['Pastel1_6'] = Pastel1_6.hex_colors
    hex_colors['Set1_4'] = Set1_4.hex_colors
    hex_colors['Set1_5'] = Set1_5.hex_colors
    hex_colors['Set1_6'] = Set1_6.hex_colors

    # Colorbrewer | Diverging
    hex_colors['RdBu_11_r'] = RdBu_11_r.hex_colors
    hex_colors['Spectral_4_r'] = Spectral_4_r.hex_colors
    hex_colors['Spectral_11_r'] = Spectral_11_r.hex_colors
    hex_colors['Spectral_7'] = Spectral_7.hex_colors
    # Colorbrewer | Sequential
    hex_colors['Reds_9'] = Reds_9.hex_colors
    hex_colors['Oranges_9'] = Oranges_9.hex_colors
    hex_colors['Blues_9'] = Blues_9.hex_colors
    hex_colors['Greens_9'] = Greens_9.hex_colors
    hex_colors['Purples_9'] = Purples_9.hex_colors

    return hex_colors