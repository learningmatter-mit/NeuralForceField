import json
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("default")

DPI = 100
LINEWIDTH = 2
FONTSIZE = 20
LABELSIZE = 18
ALPHA = 0.8
LINE_MARKERSIZE = 15 * 25
MARKERSIZE = 15
GRIDSIZE = 40
MAJOR_TICKLEN = 6
MINOR_TICKLEN = 3
TICKPADDING = 5
SECONDARY_CMAP = "inferno"

params = {
    "mathtext.default": "regular",
    "font.family": "Arial",
    "font.size": FONTSIZE,
    "axes.labelsize": LABELSIZE,
    "axes.titlesize": FONTSIZE,
    "grid.linewidth": LINEWIDTH,
    "lines.linewidth": LINEWIDTH,
    "lines.markersize": MARKERSIZE,
    "xtick.major.size": MAJOR_TICKLEN,
    "xtick.minor.size": MINOR_TICKLEN,
    "xtick.major.pad": TICKPADDING,
    "xtick.minor.pad": TICKPADDING,
    "ytick.major.size": MAJOR_TICKLEN,
    "ytick.minor.size": MINOR_TICKLEN,
    "ytick.major.pad": TICKPADDING,
    "ytick.minor.pad": TICKPADDING,
    "axes.linewidth": LINEWIDTH,
    "legend.fontsize": LABELSIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "ytick.major.width": LINEWIDTH,
    "xtick.major.width": LINEWIDTH,
    "ytick.minor.width": LINEWIDTH,
    "xtick.minor.width": LINEWIDTH,
}
plt.rcParams.update(params)


def hex_to_rgb(value: str) -> tuple:
    """
    Converts hex to rgb colours

    Parameters
    ----------
    value: string of 6 characters representing a hex colour

    Returns
    ----------
    tuple of 3 integers representing the RGB values
    """

    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value: list):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)

    Parameters
    ----------
    value: list of 3 integers representing the RGB values

    Returns
    ----------
    list of 3 floats representing the RGB values
    """

    return [v / 256 for v in value]


def get_continuous_cmap(hex_list: List[str], float_list: Optional[List[float]] = None) -> matplotlib.colors.Colormap:
    """
    Creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    Colormap
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap("j_cmap", segmentdata=cdict, N=256)
    return cmp


# colors taken from Johannes Dietschreit's script and interpolated with correct lightness and Bezier
# http://www.vis4.net/palettes/#/100|s|fce1a4,fabf7b,f08f6e,d12959,6e005f|ffffe0,ff005e,93003a|1|1
hex_list: List[str]
dir_name = Path(__file__).parent

with open(dir_name / "config/mpl_settings.json", "r") as f:
    hex_list = json.load(f)["plot_colors"]

cmap = get_continuous_cmap(hex_list)
colors = list(reversed(["#fce1a4", "#fabf7b", "#f08f6e", "#d12959", "#6e005f"]))
