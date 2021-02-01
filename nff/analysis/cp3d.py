"""
Tools for analyzing conformer-based model predictions.
"""

import os
import pickle
import random
import logging
import json

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from rdkit import Chem


from nff.utils import fprint
from nff.data.features import get_e3fp

LOGGER = logging.getLogger()
LOGGER.disabled = True

FP_FUNCS = {"e3fp": get_e3fp}


def get_pred_files(model_path):
    """
    Get pickle files with model predictions, fingerprints,
    learned weights, etc.
    Args:
        model_path (str): path where the prediction files are
            saved
    Returns:
        pred_files (list[str]): prediction file paths
    """

    pred_files = []
    for file in os.listdir(model_path):
        # should have the form <split>_pred_<metric>.pickle
        # or pred_<metric>.pickle
        splits = ["train", "val", "test"]
        starts_split = any([file.startswith(f"{split}_pred")
                            for split in splits])
        starts_pred = any([file.startswith(f"pred")
                           for split in splits])
        if (not starts_split) and (not starts_pred):
            continue
        if not file.endswith("pickle"):
            continue

        pred_files.append(os.path.join(model_path, file))

    return pred_files


def load_preds(pred_files):
    """
    Load the predictions from the predcition files
    and put them in a dictionary.
    Args:
        pred_files (list[str]): prediction file paths
    Returns:
        pred (dic): dictionary of the form {file_name:
            predictions} for each file name.
    """
    pred = {}
    for file in tqdm(pred_files):
        with open(file, "rb") as f_open:
            this_pred = pickle.load(f_open)
        name = file.split("/")[-1].split(".pickle")[0]
        pred[name] = this_pred

    return pred


def get_att_type(dic):
    """
    Figure out what kind of attention and how many heads were used.
    Args:
        dic (dict): prediction dictionary
    Returns:
        num_heads (int): number of attention heads
        is_linear (bool): whether linear attention was used (as opposed
            to pair-wise).
    """
    num_weights_list = []
    num_confs_list = []

    for sub_dic in dic.values():
        num_learned_weights = sub_dic['learned_weights'].shape[0]
        num_confs = sub_dic['boltz_weights'].shape[0]

        if num_learned_weights in num_weights_list:
            continue

        if num_confs == 1:
            continue

        num_weights_list.append(num_learned_weights)
        num_confs_list.append(num_confs)

        if len(num_confs_list) == 2:
            break

    is_linear = ((num_weights_list[1] / num_weights_list[0])
                 == (num_confs_list[1] / num_confs_list[0]))
    if is_linear:
        num_heads = int(num_weights_list[0] / num_confs_list[0])
    else:
        num_heads = int((num_weights_list[0] / num_confs_list[0] ** 2))

    return num_heads, is_linear


def annotate_confs(dic):
    """
    Annotate conformers with "head_weights" (the attention weights assigned
    to each conformer, split up by head, and also summed over conformer pairs
    if using pairwise attention), "max_weight_conf" (the conformer with the
    highest attention weight of any conformer among all heads), and
    "max_weight_head" (the head that gave this conformer its weight)/
    Args:
        dic (dict): prediction dictionary
    Returns:
        None
    """
    num_heads, is_linear = get_att_type(dic)
    for sub_dic in dic.values():
        num_confs = sub_dic['boltz_weights'].shape[0]
        if is_linear:
            split_sizes = [num_confs] * num_heads
        else:
            split_sizes = [num_confs ** 2] * num_heads

        learned = torch.Tensor(sub_dic['learned_weights'])
        head_weights = torch.split(learned, split_sizes)
        # if it's not linear, sum over conformer pairs to
        # get the average importance of each conformer
        if not is_linear:
            head_weights = [i.reshape(num_confs, num_confs).sum(0)
                            for i in head_weights]

        # the conformers with the highest weight, according to each
        # head
        max_weight_confs = [head_weight.argmax().item()
                            for head_weight in head_weights]
        # the highest conformer weight assigned by each head
        max_weights = [head_weight.max()
                       for head_weight in head_weights]
        # the head that gave out the highest weight
        max_weight_head = np.argmax(max_weights)
        # the conformer with the highest of all weights
        max_weight_conf = max_weight_confs[max_weight_head]

        sub_dic["head_weights"] = {i: weights.tolist() for i, weights in
                                   enumerate(head_weights)}
        sub_dic["max_weight_conf"] = max_weight_conf
        sub_dic["max_weight_head"] = max_weight_head


def choices_from_pickle(paths):
    """
    Get conformer choices as RDKit mols from pickle paths.
    Args:
        paths (list[str]): conformer path for each of the two
            molecules being compared.
    Returns:
        fp_choices (list[list[rdkit.Chem.rdchem.Mol]]):
            RDKit mol choices for each of the two molecules.
    """
    fps_choices = []
    for path in paths:
        with open(path, "rb") as f:
            dic = pickle.load(f)
        choices = [sub_dic["rd_mol"] for sub_dic in dic["conformers"]]
        for mol in choices:
            mol.SetProp("_Name", "test")
        fps_choices.append(choices)
    return fps_choices


def funcs_for_external(external_fp_fn,
                       summary_path,
                       rd_path):
    """
    If requesting an external method to get and compare
    fingerprints, then use this function to get a dictionary
    of pickle paths for each smiles, and the external
    fingerprinting function.
    Args:
        external_fp_fn (str): name of the fingerprinting function
            you want to use
        summary_path (str): path of the file with the summary
            dictionary of species properties, their pickle
            paths, etc.
        rd_path (str): path to the folder that has all your
            pickles with RDKit mols.
    Returns:
        pickle_dic (dict): dictionary of the form {smiles:
            full_pickle_path} for each smiles
        func (callable): fingerprinting function
    """

    func = FP_FUNCS[external_fp_fn]
    with open(summary_path, "r") as f:
        summary = json.load(f)
    pickle_dic = {}
    for key, sub_dic in summary.items():
        pickle_path = sub_dic.get("pickle_path")
        if pickle_path is None:
            continue
        pickle_dic[key] = os.path.join(rd_path, pickle_path)
    return pickle_dic, func


def sample_species(dic, classifier, max_samples):
    """
    Sample species to compare to each other.
    Args:
        dic (dict): prediction dictionary
        classifier (bool): whether your model is a classifier
        max_samples (int): maximum number of pairs to compare
    Returns:
        sample_dics (dict): dictionary with different sampling
            methods as keys, and the corresponding sampled species
            as values.
    """
    if not classifier:
        # if it's not a classifier, you'll just randomly sample
        # different species pairs and compare their fingerprints
        keys = list(dic.keys())
        samples = [np.random.choice(keys, max_samples),
                   np.random.choice(keys, max_samples)]
        sample_dics = {"random_mols": samples}
    else:
        # if it is a classifier, you'll want to compare species
        # that are both hits, both misses, or one hit and one miss

        pos_keys = [smiles for smiles, sub_dic in dic.items()
                    if sub_dic['true'] == 1]
        neg_keys = [smiles for smiles, sub_dic in dic.items()
                    if sub_dic['true'] == 0]

        intra_pos = [np.random.choice(pos_keys, max_samples),
                     np.random.choice(pos_keys, max_samples)]
        intra_neg = [np.random.choice(neg_keys, max_samples),
                     np.random.choice(neg_keys, max_samples)]
        inter = [np.random.choice(pos_keys, max_samples),
                 np.random.choice(neg_keys, max_samples)]

        sample_dics = {"intra_pos": intra_pos,
                       "intra_neg": intra_neg,
                       "inter": inter}
    return sample_dics


def calc_sim(dic,
             smiles_0,
             smiles_1,
             func,
             pickle_dic,
             conf_type,
             fp_kwargs):
    """
    Calculate the similatiy between conformers of two different species.
    Args:
        dic (dict): prediction dictionary
        smiles_0 (str): first SMILES string
        smiles_1 (str): second SMILES string
        external_fp_fn (str): name of external fingerprinting function
        func (callable): actual external fingerprinting function
        pickle_dic (dict): dictionary of the form {smiles:
            full_pickle_path} for each smiles
        conf_type (str): whether you're comparing conformers picked
            randomly for each species or based on their attention weight.
        fp_kwargs (dict): any keyword arguments you may need for your
            fingerprinting function.
    Returns:
        sim (float): cosine similarity between two conformers, one from
            each species.

    """

    sub_dic_0 = dic[smiles_0]
    sub_dic_1 = dic[smiles_1]

    if func is not None:
        paths = [pickle_dic[smiles_0], pickle_dic[smiles_1]]
        fp_0_choices, fp_1_choices = choices_from_pickle(paths)
    else:
        fp_0_choices = sub_dic_0["conf_fps"]
        fp_1_choices = sub_dic_1["conf_fps"]

    if conf_type == "att":

        conf_0_idx = sub_dic_0["max_weight_conf"]
        conf_1_idx = sub_dic_1["max_weight_conf"]

        fp_0 = fp_0_choices[conf_0_idx]
        fp_1 = fp_1_choices[conf_1_idx]

    elif conf_type == "random":
        fp_0 = random.choice(fp_0_choices)
        fp_1 = random.choice(fp_1_choices)

    fps = [fp_0, fp_1]
    for j, fp in enumerate(fps):
        if fp_kwargs is None:
            fp_kwargs = {}
        if isinstance(fp, Chem.rdchem.Mol):
            fps[j] = func(fp, **fp_kwargs)

    sim = cos_sim(fps[0].reshape(1, -1),
                  fps[1].reshape(1, -1)).item()

    return sim


def attention_sim(dic,
                  max_samples,
                  classifier,
                  seed,
                  external_fp_fn=None,
                  summary_path=None,
                  rd_path=None,
                  fp_kwargs=None):
    """
    Calculate similarities of the conformer fingerprints of different
    pairs of species.
    Args:
        dic (dict): prediction dictionary
        max_samples (int): maximum number of pairs to compare
        classifier (bool): whether your model is a classifier
        seed (int): random seed
        external_fp_fn (str, optional): name of the fingerprinting
            function you want to use. If none is provided then the model's
            generated fingerprint will be used.
        summary_path (str, optional): path of the file with the summary
            dictionary of species properties, their pickle
            paths, etc.
        rd_path (str, optional): path to the folder that has all your
            pickles with RDKit mols.
        fp_kwargs (dict, optional): any keyword arguments you need
            when calling an external fingeprinter.
    Returns:
        fp_dics (dict): dictionary of the that gives similarity scores
            between random conformers for each species, and also
            between the conformers assigned the highest attention
            weight. Has the form {sample_type: {"att": float,
            "random": float}}, where sample_type describes what kind
            of species are being sampled (e.g. both hits, both misses,
            one hit and one miss, etc.)
    """

    np.random.seed(seed)
    random.seed(seed)

    # get an external fingeprinting function if asked
    if external_fp_fn is not None:
        pickle_dic, func = funcs_for_external(external_fp_fn,
                                              summary_path,
                                              rd_path)
    else:
        pickle_dic = None
        func = None

    sample_dics = sample_species(dic, classifier, max_samples)
    fp_dics = {}

    # go through each method of sampling species and calculate their
    # conformer similarities

    for key, samples in sample_dics.items():

        fp_dics[key] = {}
        conf_types = ['att', 'random']
        for conf_type in conf_types:
            fp_sims = []
            for i in tqdm(range(len(samples[0]))):
                smiles_0 = samples[0][i]
                smiles_1 = samples[1][i]
                sim = calc_sim(dic=dic,
                               smiles_0=smiles_0,
                               smiles_1=smiles_1,
                               func=func,
                               pickle_dic=pickle_dic,
                               conf_type=conf_type,
                               fp_kwargs=fp_kwargs)
                fp_sims.append(sim)

            fp_dics[key][conf_type] = np.array(fp_sims)

    return fp_dics


def analyze_data(bare_data, analysis):
    """
    Do analysis of different fingerprints (e.g. mean, standard deviation,
    std deviation of the mean). Uses a recursive method to go through
    each sub-dictionary until an array is found.
    Args:
        bare_data (dict): dictionary with bare fingerprint similarities
        analysis (dict): same form as `bare_data` but replaces arrays
            with a dictionary analyzing their properties.
    Returns:
        None
    """
    for key, val in bare_data.items():
        if isinstance(val, np.ndarray):
            analysis[key] = {"mean": np.mean(val),
                             "std": np.std(val),
                             "std_of_mean": (np.std(val)
                                             / val.shape[0] ** 0.5)}
        else:
            if key not in analysis:
                analysis[key] = {}
            analyze_data(val, analysis[key])


def report_delta(bare_dic):
    """
    For a binary task, report analysis on the difference between
        similarity among hits and similarity between hits and misses.
    Args:
        bare_dic (dict): bare dictionary of similarities
    Returns:
        None
    """
    for key, dic in bare_dic.items():
        fprint(f"Results for {key}")
        fprint("+/- indicates standard deviation of the mean")

        # attention and random differences in similarity
        delta_att = dic['intra_pos']['att'] - dic['inter']['att']
        delta_rand = dic['intra_pos']['random'] - dic['inter']['random']

        # compute mean for attention
        delta_att_mean = np.mean(delta_att)
        # std deviation on the mean
        delta_att_std = np.std(delta_att) / (len(delta_att)) ** 0.5

        # same for random
        delta_rand_mean = np.mean(delta_rand)
        delta_rand_std = np.std(delta_rand) / (len(delta_rand)) ** 0.5

        # delta delta is the difference in deltas between random and attention,
        # a measure of how much attention is learning

        delta_delta_mean = delta_att_mean - delta_rand_mean
        delta_delta_std = ((np.var(delta_att) + np.var(delta_rand)) ** 0.5
                           / (len(delta_att)) ** 0.5)

        fprint("Delta att: %.4f +/- %.4f" % (delta_att_mean, delta_att_std))
        fprint("Delta rand: %.4f +/- %.4f" % (delta_rand_mean, delta_rand_std))
        fprint("Delta delta: %.4f +/- %.4f" %
               (delta_delta_mean, delta_delta_std))
        fprint("\n")


def conf_sims_from_files(model_path,
                         max_samples,
                         classifier,
                         seed,
                         external_fp_fn=None,
                         summary_path=None,
                         rd_path=None,
                         fp_kwargs=None):
    """
    Get similarity among species according to predictions of different
    models, given a folder with all of the prediction pickles.
    Args:
        model_path (str): path to the folder where the prediction pickles
            are saved.
        max_samples (int): maximum number of pairs to compare
        classifier (bool): whether your model is a classifier
        seed (int): random seed
        external_fp_fn (str, optional): name of the fingerprinting
            function you want to use. If none is provided then the model's
            generated fingerprint will be used.
        summary_path (str, optional): path of the file with the summary
            dictionary of species properties, their pickle
            paths, etc.
        rd_path (str, optional): path to the folder that has all your
            pickles with RDKit mols.
        fp_kwargs (dict, optional): any keyword arguments you need
            when calling an external fingeprinter.
    Returns:
        analysis (dict): dictionary of the form {prediction_name:
            similarity_dic} for the name of each prediction file.
        bare_data (dict): same idea as `analysis` but with the full
            set of similarities between each molecule.
    """

    fprint("Loading pickle files...")
    pred_files = get_pred_files(model_path)
    pred = load_preds(pred_files)

    bare_data = {}

    fprint("Calculating fingerprint similarities...")

    for key in tqdm(pred):
        dic = pred[key]
        annotate_confs(dic)
        fp_dics = attention_sim(dic=dic,
                                max_samples=max_samples,
                                classifier=classifier,
                                seed=seed,
                                external_fp_fn=external_fp_fn,
                                summary_path=summary_path,
                                rd_path=rd_path,
                                fp_kwargs=fp_kwargs)
        bare_data[key] = fp_dics

        # analyze the bare data
        analysis = {}
        analyze_data(bare_data, analysis)

        if classifier:
            report_delta(bare_data)

    return analysis, bare_data


def get_scores(path, avg_metrics=['auc', 'prc-auc']):
    """
    Load pickle files that contain predictions and actual values, using
    models evaluated by different validation metrics, and use the predictions
    to calculate and save PRC and AUC scores.
    Args:
            path (str): path to the saved model folder, which contains the
                    pickle files.
            avg_metrics (list[str]): metrics to use in score averaging
    Returns:
            scores (list): list of dictionaries containing the split being
                    used, the validation metric used to get the model, and
                    the PRC and AUC scores.
    """
    files = [i for i in os.listdir(path) if i.endswith(".pickle")
             and i.startswith("pred")]
    if not files:
        return
    scores = []
    for file in files:
        with open(os.path.join(path, file), "rb") as f:
            dic = pickle.load(f)
        split = file.split(".pickle")[0].split("_")[-1]
        from_metric = file.split("pred_")[-1].split(f"_{split}")[0]

        pred = [sub_dic['pred'] for sub_dic in dic.values()]
        true = [sub_dic['true'] for sub_dic in dic.values()]

        # then it's not a binary classification problem
        if any([i not in [0, 1] for i in true]):
            return

        auc_score = roc_auc_score(y_true=true, y_score=pred)
        precision, recall, thresholds = precision_recall_curve(
            y_true=true, probas_pred=pred)
        prc_score = auc(recall, precision)

        scores.append({"split": split,
                       "from_metric": from_metric,
                       "auc": auc_score,
                       "prc": prc_score})

    if avg_metrics is None:
        avg_metrics = [score["from_metric"] for score in scores]

    all_auc = [score["auc"] for score in scores if score['from_metric']
               in avg_metrics]
    all_prc = [score["prc"] for score in scores if score['from_metric']
               in avg_metrics]
    avg_auc = {"mean": np.mean(all_auc),
               "std": np.std(all_auc)}
    avg_prc = {"mean": np.mean(all_prc),
               "std": np.std(all_prc)}
    scores.append({"from_metric": "average",
                   "auc": avg_auc,
                   "prc": avg_prc,
                   "avg_metrics": avg_metrics})

    save_path = os.path.join(path, "scores_from_metrics.json")
    with open(save_path, "w") as f:
        json.dump(scores, f, indent=4, sort_keys=True)

    return scores


def recursive_scoring(base_path, avg_metrics=['auc', 'prc-auc']):
    """
    Recursively search in a base directory to find sub-folders that
    have pickle files that can be used for scoring. Apply `get_scores`
    to these sub-folders.
    Args:
            base_path (str): base folder to search in
            avg_metrics (list[str]): metrics to use in score averaging

    Returns:
            None
    """

    files = [i for i in os.listdir(base_path) if i.endswith(".pickle")
             and i.startswith("pred")]
    if files:
        print(f"Analyzing {base_path}")
        get_scores(base_path, avg_metrics)

    for direc in os.listdir(base_path):
        direc_path = os.path.join(base_path, direc)
        if not os.path.isdir(direc_path):
            continue
        files = [i for i in os.listdir(direc_path) if i.endswith(".pickle")
                 and i.startswith("pred")]
        if files:
            print(f"Analyzing {direc_path}")
            get_scores(direc_path, avg_metrics)
            continue

        folders = [os.path.join(direc_path, i) for i in
                   os.listdir(direc_path)]
        folders = [i for i in folders if os.path.isdir(i)]

        if not folders:
            continue
        for folder in folders:
            recursive_scoring(folder)
