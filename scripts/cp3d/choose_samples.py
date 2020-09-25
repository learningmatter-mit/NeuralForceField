import random
import json
import argparse
import pdb
import sys


def fprint(msg):
    print(msg)
    sys.stdout.flush()


def gen_splits(sample_dic,
               pos_per_val,
               prop,
               test_val_size):

    pos_smiles = [smiles for smiles, sub_dic
                  in sample_dic.items() if
                  sample_dic.get(prop) == 1]
    neg_smiles = [smiles for smiles, sub_dic
                  in sample_dic.items() if
                  sample_dic.get(prop) == 0]

    random.shuffle(pos_smiles)
    random.shuffle(neg_smiles)

    val_pos = pos_smiles[:pos_per_val]
    test_pos = pos_smiles[pos_per_val: 2 * pos_per_val]

    neg_per_val = test_val_size - pos_per_val
    val_neg = neg_smiles[:neg_per_val]
    test_neg = neg_smiles[neg_per_val: 2 * neg_per_val]

    val_all = val_pos + val_neg
    test_all = test_pos + test_neg

    for smiles in sample_dic.keys():
        if smiles in val_all:
            sample_dic[smiles].update({"split": "val"})
        elif smiles in test_all:
            sample_dic[smiles].update({"split": "test"})
        else:
            sample_dic[smiles].update({"split": "train"})

    return sample_dic


def proportional_sample(summary_dic,
                        prop,
                        max_specs,
                        sample_dic_path,
                        pos_per_val,
                        test_val_size,
                        regression):

    # need to change this for if regression = True

    """
    Sample species for a dataset so that the number of positives
    and negatives is the same proportion as in the overall
    dataset.

    """

    positives = []
    negatives = []

    for smiles, sub_dic in summary_dic.items():
        value = sub_dic.get(prop)
        if value is None:
            continue
        if sub_dic.get("pickle_path") is None:
            continue
        if value == 0:
            negatives.append(smiles)
        elif value == 1:
            positives.append(smiles)

    num_neg = len(negatives)
    num_pos = len(positives)

    if max_specs is None:
        max_specs = num_neg + num_pos

    # get the number of desired negatives and positives to
    # get the right proportional sampling

    num_neg_sample = int(num_neg / (num_neg + num_pos) * max_specs)
    num_pos_sample = int(num_pos / (num_neg + num_pos) * max_specs)

    # shuffle negatives and positives and extract the appropriate
    # number of each

    random.shuffle(negatives)
    random.shuffle(positives)

    neg_sample = negatives[:num_neg_sample]
    pos_sample = positives[:num_pos_sample]

    all_samples = [*neg_sample, *pos_sample]
    sample_dic = {key: summary_dic[key]  # ["pickle_path"]
                  for key in all_samples if "pickle_path"
                  in summary_dic[key] and prop in summary_dic[key]}

    # generate train/val/test labels

    sample_dic = gen_splits(sample_dic=sample_dic,
                            pos_per_val=pos_per_val,
                            prop=prop,
                            test_val_size=test_val_size)

    with open(sample_dic_path, "w") as f:
        json.dump(sample_dic, f, indent=4, sort_keys=True)


def main(max_specs,
         prop,
         summary_path,
         sample_dic_path,
         pos_per_val,
         test_val_size,
         regression):

    with open(summary_path, "r") as f:
        summary_dic = json.load(f)

    # TO-DO:
    # 1. Sort out what to do when it's regression
    # 2. Add option to get as many positives as possible
    # instead of doing a proportional sample

    proportional_sample(summary_dic=summary_dic,
                        prop=prop,
                        max_specs=max_specs,
                        sample_dic_path=sample_dic_path,
                        pos_per_val=pos_per_val,
                        test_val_size=test_val_size,
                        regression=regression)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_specs', type=int, default=None,
                        help=("Maximum number of species to use in your "
                              "dataset. No limit if max_specs isn't "
                              "specified."))
    parser.add_argument('--prop', type=str, default=None)
    parser.add_argument('--summary_path', type=str)
    parser.add_argument('--sample_dic_path', type=str)
    parser.add_argument('--pos_per_val', type=int)
    parser.add_argument('--test_val_size', type=int)

    parser.add_argument('--regression', action='store_true',
                        help=("Specify regression instead of classification. "
                              "In this case you don't need to supply `prop`, "
                              "and the subsampling of the dataset will be done "
                              "randomly."))

    arguments = parser.parse_args()

    try:
        main(**arguments.__dict__)
    except Exception as e:
        fprint(e)
        pdb.post_mortem()
