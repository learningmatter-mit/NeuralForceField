import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
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
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": LABELSIZE,
    "legend.fontsize": LABELSIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    'ytick.major.width': LINEWIDTH,
    'xtick.major.width': LINEWIDTH,
    'ytick.minor.width': LINEWIDTH,
    'xtick.minor.width': LINEWIDTH,

}
plt.rcParams.update(params)


def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values"""
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values"""
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map"""
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap("j_cmap", segmentdata=cdict, N=256)
    return cmp


# colors taken from Johannes Dietschreit's script and interpolated with correct lightness and Bezier
# http://www.vis4.net/palettes/#/100|s|fce1a4,fabf7b,f08f6e,d12959,6e005f|ffffe0,ff005e,93003a|1|1
hex_list = [
    "#fce1a4",
    "#fcdea1",
    "#fcdc9e",
    "#fcda9b",
    "#fcd799",
    "#fcd496",
    "#fbd294",
    "#fbcf91",
    "#fbcd8f",
    "#fbca8d",
    "#fbc88b",
    "#fac589",
    "#fac387",
    "#fac085",
    "#f9be83",
    "#f9bb82",
    "#f8b980",
    "#f8b67e",
    "#f8b47d",
    "#f7b17b",
    "#f7ae7a",
    "#f6ac79",
    "#f6a977",
    "#f5a776",
    "#f5a475",
    "#f4a274",
    "#f49f73",
    "#f39c72",
    "#f29a71",
    "#f29770",
    "#f19470",
    "#f1926f",
    "#f08f6e",
    "#ef8d6d",
    "#ee8a6d",
    "#ed876c",
    "#ec856c",
    "#eb826b",
    "#ea806b",
    "#e97d6a",
    "#e87b6a",
    "#e77869",
    "#e67669",
    "#e57368",
    "#e37168",
    "#e26f67",
    "#e16c67",
    "#df6a66",
    "#de6766",
    "#dd6566",
    "#db6365",
    "#da6065",
    "#d85e64",
    "#d65c64",
    "#d55964",
    "#d35763",
    "#d25563",
    "#d05263",
    "#ce5063",
    "#cc4e62",
    "#cb4c62",
    "#c94962",
    "#c74761",
    "#c54561",
    "#c34361",
    "#c14061",
    "#bf3e61",
    "#bd3c60",
    "#bb3a60",
    "#b93860",
    "#b73660",
    "#b53360",
    "#b33160",
    "#b12f5f",
    "#af2d5f",
    "#ac2b5f",
    "#aa295f",
    "#a8275f",
    "#a5255f",
    "#a3235f",
    "#a1215f",
    "#9e1f5f",
    "#9c1d5f",
    "#9a1b5f",
    "#97195e",
    "#95175e",
    "#92155e",
    "#8f135e",
    "#8d115e",
    "#8a0f5e",
    "#880d5e",
    "#850b5e",
    "#82095e",
    "#7f075f",
    "#7d055f",
    "#7a045f",
    "#77035f",
    "#74025f",
    "#71015f",
    "#6e005f",
]
cmap = get_continuous_cmap(hex_list)
colors = list(reversed(["#fce1a4", "#fabf7b", "#f08f6e", "#d12959", "#6e005f"]))
