"""
Tools for computing and plotting the ROCE values of different classification models
at different enrichment factors.
"""

import copy
import pickle
import json
import math

import argparse
from matplotlib import pyplot as plt
from matplotlib import rcParams

import numpy as np

from nff.utils import read_csv


# height of each ROCE bar slice in the plots, normalized
# to max value of all bars in the plot
BAR_HEIGHT = 0.02

# width taken up by each category in a bar plot
DELTA = 0.2

# keys for specifying text attributes
TEXT_KEYS = ['fontsize']


# use the same defaults as in iPython notebooks
# to avoid an unhappy surprise after testing your plots
# in a notebook

rcParams['figure.figsize'] = (6.0, 4.0)
rcParams['font.size'] = 10
rcParams['savefig.dpi'] = 72
rcParams['figure.subplot.bottom'] = 0.125


def compute_roce(fpr, preds, real):
    """
    Calculate the ROC enrichment (ROCE) score - i.e., the y value divided by
    the x value on an ROC curve, at a fixed predetermined x value (i.e.
    a fixed false positive rate).
    Args:
        fpr (float): pre-set false positive rate
        preds (np.array): model predictions for each species
        real (np.array): real class for each species (hit or miss)
    Returns:
        roce (np.array): ROCE at the specified fpr

    """

    # number of decoys (misses) in the data set
    n_dec_tot = len(real) - sum(real)

    # how many decoys must be identified by the model
    # to get the pre-set fpr. We use `math.ceil`
    # so that n_dec_chosen is always at least 1, otherwise
    # we could end up with no predictions
    n_dec_chosen = math.ceil(n_dec_tot * fpr)

    # sort the predictions from highest to lowest
    sort_idx = np.argsort(-preds)
    # get the real values of these sorted predictions
    sorted_real = real[sort_idx]

    # how many of the top predictions to take
    # such that `n_dec_chosen` are misses
    cutoff_idx = 0

    # how many decoys have been found so far
    decs_found = 0

    for i, this_real in enumerate(sorted_real):
        if this_real == 0:
            decs_found += 1

        # we've reached the pre-set number of
        # decoys to be chosen
        if decs_found == n_dec_chosen:
            cutoff_idx = i + 1
            break

    # how many positives it got right
    true_p = sum(sorted_real[:cutoff_idx])
    # how many species it said were hits and were
    # actually misses
    false_p = cutoff_idx - true_p

    # how many species it said were misses but were
    # really hits

    false_n = sum(sorted_real[cutoff_idx:])

    # how many misses it got right
    true_n = (len(sorted_real) - cutoff_idx) - false_n

    # numerator: true positive rate
    num = true_p / (true_p + false_n)

    # denominator: false positive rate
    denom = false_p / (true_n + false_p)

    # roce = tpr / fpr
    roce = num / denom

    return roce


def get_min_max(val, height):
    """
    Get the min value and max value of a horizontal slice of a bar
    plot that is used for visualization purposes when various bars
    overlap.
    Args:
        val (float): value of the ROCE
        height (float): height of the horizontal slice
    Returns:
        mini (float): min value
        maxi (float): max value
    """

    mini = val - height / 2
    maxi = val + height / 2
    return mini, maxi


def get_change(this_val, other_val, height):
    """
    Figure out how much you have to change an ROCE to make sure
    the horizontal slice of the bar plot doesn't overlap with
    another horizontal slice.
    Args:
        this_val (float): this value of the ROCE
        other_val (float): value of a different ROCE
        height (float): height of the horizontal slice
    Returns:
        change (float): how much you have to add to, or subtract
            from, the ROCE to make the slices not overlap.
    """

    # get the min and max values of the two slices
    this_min, this_max = get_min_max(this_val, height)
    other_min, other_max = get_min_max(other_val, height)

    # whether this slice extends below the other slice
    upper_overlap = other_min <= this_max <= other_max

    # whether this slice extends above the other slice
    lower_overlap = other_min <= this_min <= other_max

    # if neither, they don't overlap
    if (not upper_overlap) and (not lower_overlap):
        return 0

    # otherwise calculate the required shift of `this_val`
    if upper_overlap:
        # how much overlap there is between the two slices
        common_height = this_min - other_min
        # shift down to cover the overlap, and then again
        # by `height`
        change = -(height + common_height)

    elif lower_overlap:
        common_height = other_max - this_min
        # only shift down by the overlap
        change = -(common_height)

    return change


def remove_overlap(scores, height):
    """
    Remove overlap between all horizontal bar slices
    for each model.
    Args:
        scores (np.array): ROCE array of dimension `num_models`
            x `num_fpr`, where `num_models` is the number of
            different models and `num_fpr` is the number of
            different pre-set false positive rates chosen.
        height (float): height of the horizontal bar slices
    Returns:
        new_vals (np.array): updated version of `scores` to
            ensure no overlap in the horizontal bar slices.
    """

    # tolerance for overlap - if they overlap by this amount
    # or less, then they're considered to not overlap
    eps = 1e-5

    # number of models
    num_models = scores.shape[0]

    # number of pre-set fpr values
    num_fpr = scores.shape[1]

    # updated scores to remove overlap
    new_vals = copy.deepcopy(scores)

    # go through each model
    for model in range(num_models):
        vals = []

        # go through each fpr
        for i in range(num_fpr):
            # continue until all the changes in positions are below the
            # tolerance
            while True:
                change = 0
                # compare this value to all the other values of
                # the model that are ordered before this one
                for other_val in vals:
                    # calculate how much you have to change this value
                    change = get_change(this_val=new_vals[model, i],
                                        other_val=other_val,
                                        height=height)
                    # if you've changed it at all after comparing
                    # to another value, break
                    if abs(change) > eps:
                        break
                # update new_vals
                new_vals[model, i] += change

                # if there's no change then break
                if abs(change) < eps:
                    break
            # add this value to `vals`
            vals.append(new_vals[model, i])
    return new_vals


def parse_csv(pred_path,
              true_path,
              target):
    """
    Get the list of predicted and real values from a csv file.
    Running `predict.sh` on the results of a ChemProp calculation
    produces a csv file of the predictions of each ChemProp fold
    and a JSON file that summarizes the predictions of each fold.

    Args:
        pred_path (str): path to predicted values
        true_path (str): path to real values
        target (str): name of property you're predicting
    Returns:
        pred (list[np.array]): the predictions of this model.
            Given as a list of length 1 that contains an
            array of length `num_species` (number of species).
            Given in this way to be consistent with `parse_json`
            below.
        real (list[np.array]): same as `pred` but with the real
            values.
    """

    pred_dic = read_csv(pred_path)
    pred = np.array(pred_dic[target])

    real_dic = read_csv(true_path)
    real = np.array(real_dic[target])

    return [pred], [real]


def parse_json(pred_path,
               target,
               split):
    """
    Get the list of predicted and real values from a JSON file.
    Running `predict.sh` on the results of a ChemProp calculation
    produces a csv file of the predictions of each ChemProp fold
    and a JSON file that summarizes the predictions of each fold.

    The contents of the JSON file have the following form:

    {"0": {
        "sars_cov_two_cl_protease_active": {
            "test": {
                "pred": [0.1, 0.2, ...],
                "real": [0, 0, 1, ...]},
            "train": {
                "pred": [0.1, 0.2, ...],
                "real": [0, 0, 1, ...]}}},
    "1": {...}}

    Args:
        pred_path (str): path to predicted values
        target (str): name of property you're predicting
        split (str): name of the split (i.e. test, train,
            or val).
    Returns:
        pred (list[np.array]): the predictions of versions of this
             model with different seeds. Each item of the list
             is the set of predictions from a different model.
        real (list[np.array]): same as `pred` but with the real
            values.
    """

    with open(pred_path, 'r') as f_open:
        pred_dic = json.load(f_open)

    # int keys for different seeds
    int_keys = list([i for i in pred_dic.keys()
                     if i.isdigit()])

    # get the predictions of each seed
    preds = []
    reals = []

    for key in int_keys:
        sub_preds = np.array(pred_dic[key][target][split]["pred"])
        preds.append(sub_preds)

        sub_reals = np.array(pred_dic[key][target][split]["true"])
        reals.append(sub_reals)

    return preds, reals


def parse_pickle(path):
    """
    Get the list of predicted and real values from a pickle file,
    which is produced by running `make_fps` on the results of
    a 3D or 4D model.  The contents of the pickle file have
    the following form:

    {"CCN": {"true": 0.1, "pred": 0},
     "C#N": {"true": 0.1, "pred": 0},
    ...}

    Args:
        path (str): path to predicted values
    """

    with open(path, "rb") as f_open:
        dic = pickle.load(f_open)

    real = np.array([sub_dic['true'] for sub_dic in dic.values()])
    pred = np.array([sub_dic['pred'] for sub_dic in dic.values()])

    return [pred], [real]


def get_all_preds(true_path,
                  pred_paths,
                  target,
                  split):
    """
    Get all predictions from various different versions of a model
    (e.g. different seeds of a ChemProp model).
    Args:
        true_path (str): path to real values
        pred_paths (list[str]): paths to predicted values
        target (str): name of property you're predicting
        split (str): name of the split (i.e. test, train,
            or val).
    Returns:
        preds (np.array): prediction array of shape `num_model_versions`
            x `num_species`, where `num_model_versions` is the number
            of versions of this model and `num_species` is the number
            of species in the dataset.
        reals (np.array): same as `preds` but for the real values.
    """

    preds = []
    reals = []

    for path in pred_paths:
        if path.endswith("csv"):
            these_preds, these_real = parse_csv(pred_path=path,
                                                true_path=true_path,
                                                target=target)

        elif path.endswith("json"):
            these_preds, these_real = parse_json(pred_path=path,
                                                 target=target,
                                                 split=split)
        elif path.endswith("pickle"):
            these_preds, these_real = parse_pickle(path)
        else:
            raise Exception(f"Suffix of {path} not recognized")

        preds += these_preds
        reals += these_real

    preds = np.stack(preds)
    reals = np.stack(reals)

    return preds, reals


def get_mean_roce(true_path,
                  pred_paths,
                  target,
                  split,
                  fpr_vals):
    """
    Get mean ROCE scores from various different versions of a model
    (e.g. different seeds of a ChemProp model).
    Args:
        true_path (str): path to real values
        pred_paths (list[str]): paths to predicted values
        target (str): name of property you're predicting
        split (str): name of the split (i.e. test, train,
            or val).
        fpr_vals (np.array): pre-set false positive rates
    Returns:
        mean_roce (np.array): array of ROCE values at each fpr,
            averaged over the different versions of the model.
    """

    all_preds, all_reals = get_all_preds(true_path=true_path,
                                         pred_paths=pred_paths,
                                         target=target,
                                         split=split)
    roces = []
    for pred, real in zip(all_preds, all_reals):
        roce = np.array([compute_roce(fpr, pred, real)
                         for fpr in fpr_vals])
        roces.append(roce)

    mean_roce = np.stack(roces).mean(axis=0)

    return mean_roce


def add_model_roces(plot_dic):
    """
    Add the mean ROCEs of each model type to `plot_dic`,
    the dictionary with information about the different plots.
    Args:
        plot_dic (dict): the dictionary with information about
            the different plots.
    Returns:
        plot_dic (dict): updated version with mean ROCE scores
    """

    base_info = plot_dic["base_info"]
    model_dics = plot_dic["model_dics"]

    true_path = base_info["true_path"]
    target = base_info["target"]
    split = base_info["split"]
    fpr_vals = base_info["fpr_vals"]

    for i, model_dic in enumerate(model_dics):
        mean_roce = get_mean_roce(
            true_path=true_path,
            pred_paths=model_dic["pred_paths"],
            target=target,
            split=split,
            fpr_vals=fpr_vals)
        model_dics[i]["roce"] = mean_roce

    return plot_dic


def vals_for_plot(plot_dic):
    """
    Compute or create some items that will be needed in the final plot.
    Args:
        plot_dic (dict): the dictionary with information about
            the different plots.
    Returns:
        roce_scores (np.array): mean ROCE scores for each model type
            and each fpr value. Has dimension `num_models` x
            `num_fpr`, where `num_models` is the number of different
            model types and `num_fpr` is the number of preset
            false positive rates.
        roce_no_overlap (np.array): same as `roce_scores`, but
            adjusted so that the horizontal bar slices in each
            model don't overlap with each other.
        fpr_colors (list[str]): list of the colors assigned
            to the bars at each fpr rate.
        labels (list[str]): labels given to each of the different models.
        bar_height (float): height of the horizontal bar slices.

    """

    # add the ROCEs
    plot_dic = add_model_roces(plot_dic)
    model_dics = plot_dic["model_dics"]

    # stack the values from the different models
    roce_scores = np.stack([dic["roce"] for dic in model_dics])
    # scale `BAR_HEIGHT` by the largest value in the plot
    bar_height = BAR_HEIGHT * roce_scores.max()
    # calculate the roces with no overlap
    roce_no_overlap = remove_overlap(roce_scores, bar_height)

    # get the default cycle colors
    fpr_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # set the labels equal to the plot names, but replace every space
    # with a new line to avoid overlapping labels
    plot_names = [dic["plot_name"] for dic in model_dics]
    labels = [l.replace(" ", "\n") for l in plot_names]

    return roce_scores, roce_no_overlap, fpr_colors, labels, bar_height


def base_plot(roce_scores,
              roce_no_overlap,
              labels,
              fpr_colors,
              bar_height):
    """
    Make the basic ROCE plot without any extra features, label sizes, axis
    limits, etc.
    Args:
        roce_scores (np.array): mean ROCE scores for each model type
            and each fpr value. Has dimension `num_models` x
            `num_fpr`, where `num_models` is the number of different
            model types and `num_fpr` is the number of preset
            false positive rates.
        roce_no_overlap (np.array): same as `roce_scores`, but
            adjusted so that the horizontal bar slices in each
            model don't overlap with each other.
        labels (list[str]): labels given to each of the different models.
        fpr_colors (list[str]): list of the colors assigned
            to the bars at each fpr rate.
        bar_height (float): height of the horizontal bar slices.
    Returns:
        axis (maplotlib.axes._subplots.AxesSubplot): axis of the base
            plot, to be used later for adding additional features to
            the plot.
    """

    # make the axis and set the x tick labels
    _, axis = plt.subplots()
    axis.set_xticklabels(labels)

    # make the bar plots for the different models, and assign colors
    # based on the fpr value being used

    for i in range(roce_scores.shape[1]):
        axis.bar(labels, roce_scores[:, i], color=fpr_colors[i])

    # plot the horizontal slices using `roce_no_overlap`
    # go through each value of fpr

    for j in range(roce_no_overlap.shape[1]):

        # start is where the slice starts
        start = -0.4

        # go through each model
        for i in range(roce_no_overlap.shape[0]):
            # the adjusted roce value
            new_perform = roce_no_overlap[i, j]
            # the end of the horizontal slice
            end = start + 0.8
            # the width
            interval = end - start

            # plot the new ROCE value from `start` to `end`,
            # with label `_nolegend_` to make sure these don't
            # get labeled in the legend

            x_range = np.arange(start, end, interval / 100)
            y_vals = np.array([new_perform] * len(x_range))

            plt.plot(x_range, y_vals,
                     '-',
                     color=fpr_colors[j],
                     linewidth=3,
                     label='_nolegend_')

            # add black lines at +/- bar_height / 2
            plt.plot(x_range, (y_vals - bar_height / 2),
                     '-',
                     color='black',
                     linewidth=1,
                     label='_nolegend_')
            plt.plot(x_range, (y_vals + bar_height / 2),
                     '-',
                     color='black',
                     linewidth=1,
                     label='_nolegend_')

            # add `DELTA` to `start` as you continue left to
            # right in the plot
            start = end + DELTA

    return axis


def set_plot_ylim(max_scale,
                  roce_no_overlap,
                  bar_height):
    """
    Set the y limits for the plot.
    Args:
        max_scale (float): the maximum y limit is given by
            the maximum value of the ROCEs x `max_scale`
        roce_no_overlap (np.array): ROCEs adjusted to have
            no overlap.
        bar_height (float): height of the horizontal bar slices
    Returns:
        ylim (list): y-limits given by [min_y, max_y]
    """

    min_score = roce_no_overlap.min()
    max_score = roce_no_overlap.max()
    min_scale = max_scale if (min_score < 0) else (1 / max_scale)

    ylim = [min([min_score * min_scale, 0]) -
            bar_height, max_score * max_scale]
    plt.ylim(ylim)

    return ylim


def set_tick_sizes(x_axis_dic,
                   y_axis_dic):
    """
    Set plot tick sizes.
    Args:
        x_axis_dic (dict): dictionary with information about
            the x-axis.
        y_axis_dic (dict): dictionary with information about
            the y-axis.
    Returns:
        None
    """

   # x-axis tick font size
    if "ticks" in x_axis_dic:
        tick_dic = x_axis_dic["ticks"]
        if "fontsize" in tick_dic:
            plt.rc('xtick', labelsize=tick_dic["fontsize"])

    # y-axis tick font size
    if "ticks" in y_axis_dic:
        tick_dic = y_axis_dic["ticks"]
        if "fontsize" in tick_dic:
            plt.rc('ytick', labelsize=tick_dic["fontsize"])


def label_plot(fpr_vals,
               legend_dic,
               x_axis_dic,
               y_axis_dic,
               axis):
    """
    Add various labels to the plot.
    Args:
        fpr_vals (np.array): pre-set false positive rates
        legend_dic (dict): dictionary with information about
            the legend.
        x_axis_dic (dict): dictionary with information about
            the x-axis.
        y_axis_dic (dict): dictionary with information about
            the y-axis.
        axis (maplotlib.axes._subplots.AxesSubplot): axis of the
            plot.
    Returns:
        None
    """

    # legend
    fpr_pct = [(use_val * 100) for use_val in fpr_vals]
    fpr_str = [("%.1f" % val) if (val < 1) else ("%d" % val)
               for val in fpr_pct]

    kwargs = {key: legend_dic[key] for key in
              [*TEXT_KEYS, 'loc', 'ncol'] if key in legend_dic}
    if legend_dic.get("use_legend", True):
        plt.legend([f'{string}%' for string in fpr_str],
                   **kwargs)

    # y-axis font size and label
    ylabel_kwargs = {}
    if 'labels' in y_axis_dic:
        label_dic = y_axis_dic['labels']
        if 'fontsize' in label_dic:
            ylabel_kwargs["fontsize"] = label_dic["fontsize"]

    plt.ylabel("ROCE", **ylabel_kwargs)

    # x-axis label font sizes
    if 'labels' in x_axis_dic:
        label_dic = x_axis_dic['labels']
        if 'fontsize' in label_dic:
            for label in axis.get_xticklabels():
                label.set_fontsize(label_dic['fontsize'])


def decorate_plot(labels,
                  ylim,
                  axis,
                  dividers=None,
                  texts=None):
    """
    Add various "decorations" to the plot - such as dividers between
    different model categories, text on the plot, etc.
    Args:
        labels (list[str]): labels given to each of the different models
        ylim (list): y-limits given by [min_y, max_y]
        axis (maplotlib.axes._subplots.AxesSubplot): axis of the
            plot.
        dividers (list[str], optional): names of model categories after
            which to put a vertical dividing line.
        texts (list[dict], optional): list of dictionaries with
            information about any text you want to place on the plot.
    Returns:
        None
    """

    max_x = len(labels)
    x_range = np.arange(-0.5, max_x, max_x/100)
    plt.plot(x_range, [0] * len(x_range),
             '-',
             color='black',
             linewidth=1,
             label='_nolegend_',
             zorder=-10)

    # add any dividers
    if dividers is not None:
        locs = []
        for divider in dividers:
            loc = labels.index(divider.replace(" ", "\n")) + 2.5 * DELTA
            locs.append(loc)
        plt.vlines(locs,
                   ylim[0],
                   ylim[1],
                   linestyles='--',
                   color='black')

    # add any text
    if texts is not None:
        for item in texts:
            text = item['text']
            pos = item['position']
            kwargs = {key: item[key] for key in TEXT_KEYS
                      if key in item}

            plt.text(*pos, text,
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axis.transAxes,
                     **kwargs)


def save_plot(save_path):
    """
    Save plot.
    Args:
        save_path (str): path to save file to
    Returns:
        None
    """

    if save_path is None:
        return

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved figure to {save_path}")


def plot(plot_dic):
    """
    Make complete plot.
    Args:
        plot_dic (dict): the dictionary with information about
            the different plots.
    Returns:
        roce_scores (np.array): mean ROCE scores for each model type
            and each fpr value. Has dimension `num_models` x
            `num_fpr`, where `num_models` is the number of different
            model types and `num_fpr` is the number of preset
            false positive rates.
        labels (list[str]): labels given to each of the different models.
        fpr_vals (np.array): pre-set false positive rates
    """

    # basic information that applies to all models in the plot
    base_info = plot_dic["base_info"]
    # information about the plot display
    plot_info = plot_dic["plot_info"]

    # get ROCE scores and other values needed for the plot
    (roce_scores, roce_no_overlap,
        fpr_colors, labels, bar_height) = vals_for_plot(plot_dic=plot_dic)

    # set tick sizes - this has to come before making the plot
    x_axis_dic = plot_info.get("x_axis", {})
    y_axis_dic = plot_info.get("y_axis", {})

    set_tick_sizes(x_axis_dic=x_axis_dic,
                   y_axis_dic=y_axis_dic)

    # make the base plot
    axis = base_plot(roce_scores=roce_scores,
                     roce_no_overlap=roce_no_overlap,
                     labels=labels,
                     fpr_colors=fpr_colors,
                     bar_height=bar_height)

    # add labels
    label_plot(fpr_vals=base_info["fpr_vals"],
               legend_dic=plot_info.get("legend", {}),
               x_axis_dic=x_axis_dic,
               y_axis_dic=y_axis_dic,
               axis=axis)

    # set the y limits
    ylim = set_plot_ylim(max_scale=plot_info.get("max_height_scale", 1.2),
                         roce_no_overlap=roce_no_overlap,
                         bar_height=bar_height)

    # add decorations
    decorate_plot(labels=labels,
                  ylim=ylim,
                  axis=axis,
                  dividers=plot_info.get("dividers"),
                  texts=plot_info.get("texts"))

    # save and show
    save_plot(save_path=plot_info.get("save_path"))

    plt.show()

    fpr_vals = plot_dic["base_info"]["fpr_vals"]
    return roce_scores, labels, fpr_vals


def get_perform_info(fprs,
                     roce_scores,
                     labels):
    """
    Summarize the information about model performances so it
    can be saved in a JSON.

    Args:
        fprs (np.array): pre-set false positive rates
        roce_scores (np.array): mean ROCE scores for each model type
            and each fpr value. Has dimension `num_models` x
            `num_fpr`, where `num_models` is the number of different
            model types and `num_fpr` is the number of preset
            false positive rates.
        labels (list[str]): labels given to each of the different models.

    Returns:
        info (list[dict]): list of dictionaries, each of which says the FPR
            value and the ROCE scores/model names of each model.
    """

    info = []

    for i, fpr in enumerate(fprs):
        scores = roce_scores[:, i].reshape(-1)
        sort_idx = np.argsort(-scores)
        sort_scores = scores[sort_idx].tolist()
        sort_labels = np.array(labels)[sort_idx].tolist()

        score_list = [{"rank": i + 1,
                       "model": sort_labels[i].replace("\n", " "),
                       "roce": score}
                      for i, score in enumerate(sort_scores)]

        this_info = {"fpr": fpr,
                     "scores": score_list}
        info.append(this_info)

    return info


def plot_all(plot_dics):
    """
    Make complete set of plot from various plot dictionaries.
    Args:
        plot_dics (list[dict]): different dictionaries with information
            about the different plots.
    Returns:
        roce (list): ROCE scores and labels of each model from each of the plots
    """

    roces = []

    for plot_dic in plot_dics:
        roce_scores, labels, fprs = plot(plot_dic)
        info = get_perform_info(fprs=fprs,
                                roce_scores=roce_scores,
                                labels=labels)
        roces.append(info)

    return roces


def main():
    """
    Make plots from the command line by specifying the path to the JSON
    file with plot information.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with plot information. "
                              "Please see config/plot_info.json for an "
                              "example."))
    parser.add_argument('--save_path', type=str,
                        help=("Path to JSON file with saved ROCE scores."),
                        default='roce.json')

    args = parser.parse_args()
    config_file = args.config_file
    with open(config_file, "r") as f_open:
        plot_dics = json.load(f_open)

    roces = plot_all(plot_dics=plot_dics)
    save_path = args.save_path
    with open(save_path, 'w') as f_open:
        json.dump(roces, f_open, indent=4, sort_keys=True)

    print(f"Saved ROCE score information to {save_path}")


if __name__ == "__main__":
    main()
