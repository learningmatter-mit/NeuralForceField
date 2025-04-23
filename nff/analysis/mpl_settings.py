import json
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("default")

dir_name = Path(__file__).parent

DPI = 300
LINEWIDTH = 1.25
FONTSIZE = 8
LABELSIZE = 8
ALPHA = 0.8
MARKERSIZE = 25
GRIDSIZE = 20
MAJOR_TICKLEN = 4
MINOR_TICKLEN = 2
TICKPADDING = 3
SECONDARY_CMAP = "inferno"

custom_settings = {
    "mathtext.default": "regular",
    "font.family": "Arial",
    "font.size": FONTSIZE,
    "axes.labelsize": LABELSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.linewidth": LINEWIDTH,
    "grid.linewidth": LINEWIDTH,
    "lines.linewidth": LINEWIDTH,
    "lines.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "axes.titlecolor": "black",
    "axes.titleweight": "bold",
    "axes.grid": False,
    "lines.markersize": MARKERSIZE,
    "xtick.major.size": MAJOR_TICKLEN,
    "xtick.minor.size": MINOR_TICKLEN,
    "xtick.major.pad": TICKPADDING,
    "xtick.minor.pad": TICKPADDING,
    "ytick.major.size": MAJOR_TICKLEN,
    "ytick.minor.size": MINOR_TICKLEN,
    "ytick.major.pad": TICKPADDING,
    "ytick.minor.pad": TICKPADDING,
    "ytick.major.width": LINEWIDTH,
    "xtick.major.width": LINEWIDTH,
    "ytick.minor.width": LINEWIDTH,
    "xtick.minor.width": LINEWIDTH,
    "legend.fontsize": LABELSIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.format": "png",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "figure.facecolor": "white",
}
plt.rcParams.update(custom_settings)


def update_custom_settings(custom_settings: dict = custom_settings) -> None:
    """Update the custom settings for Matplotlib.

    Args:
        custom_settings (dict, optional): Custom settings for Matplotlib. Defaults to
            custom_settings.
    """
    current_settings = plt.rcParams.copy()
    new_settings = current_settings | custom_settings
    plt.rcParams.update(new_settings)


def hex_to_rgb(value: str) -> list[float]:
    """Converts hex to rgb colors.

    Args:
        value (str): string of 6 characters representing a hex colour.

    Returns:
        list: length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value: list[float]) -> list[float]:
    """Converts rgb to decimal colors (i.e. divides each value by 256).

    Args:
        value (list[float]): string of 6 characters representing a hex colour.

    Returns:
        list: length 3 of RGB values
    """
    return [v / 256 for v in value]


def get_continuous_cmap(
    hex_list: list[str], float_list: list[float] | None = None
) -> mpl.colors.LinearSegmentedColormap:
    """Creates a color map that can be used in heat map figures. If float_list is not provided,
    color map graduates linearly between each color in hex_list. If float_list is provided,
    each color in hex_list is mapped to the respective location in float_list.

    Args:
        hex_list (list[str]): list of hex code strings
        float_list (list[float]): list of floats between 0 and 1, same length as hex_list. Must
            start with 0 and end with 1.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: continuous
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))
        ]
        cdict[col] = col_list
    return mpl.colors.LinearSegmentedColormap("j_cmap", segmentdata=cdict, N=256)


# Colors taken from Johannes Dietschreit's script and interpolated with correct lightness and Bezier
# http://www.vis4.net/palettes/#/100|s|fce1a4,fabf7b,f08f6e,d12959,6e005f|ffffe0,ff005e,93003a|1|1
hex_list: List[str]
with open(dir_name / "config/mpl_settings.json", "r") as f:
    hex_list = json.load(f)["plot_colors"]

cmap = get_continuous_cmap(hex_list)
colors = list(reversed(["#fce1a4", "#fabf7b", "#f08f6e", "#d12959", "#6e005f"]))
