from sigopt import Connection
import re
import os
import numpy as np
import subprocess
import json
from multiprocessing import Process
import pdb
import sys
import time
from datetime import datetime

TOKEN = "KTNMWLZQYQSNCHVHPGIWSAVXEWLEWABZAHIJOLXKWAHQDRQE"
BASE_SAVE_PATH = ("/home/saxelrod/engaging_nfs/data_from_fock/"
                  "combined_fingerprint_datasets")
BASE_CHEMPROP_PATH = "/home/saxelrod/chemprop_sigopt"
FEATS = ['mean_e3fp', 'morgan']
PROPS = ["ensembleentropy"]


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def make_expt(name, token=TOKEN):
    conn = Connection(client_token=token)
    experiment = conn.experiments().create(
        name=name,
        metrics=[dict(name='mae', objective='minimize')],
        parameters=[
            dict(name='log_dropout',
                 type='double', bounds=dict(min=-5, max=0)),
        ],
        observation_budget=20
    )

    return conn, experiment


def read_csv(path):

    dic = {}
    ordered_smiles = []

    with open(path, "r") as f:
        lines = f.readlines()
        prop_names = [name.strip() for name in lines[0].split(",")[1:]]

        for line in lines[1:]:
            smiles = line.split(",")[0].strip()
            props = [item.strip() for item in line.split(",")[1:]]
            for i in range(len(props)):
                prop = props[i]
                try:
                    prop = int(prop)
                except ValueError:
                    prop = float(prop)
                    props[i] = prop

            dic[smiles] = {prop_name: prop for prop_name, prop in
                           zip(prop_names, props)}
            ordered_smiles.append(smiles)

    return dic, ordered_smiles


def write_csv(path, dic):

    smiles_keys = sorted(list(dic.keys()))

    sub_dic_0 = dic[smiles_keys[0]]
    prop_keys = sorted(list(sub_dic_0.keys()))

    text = "smiles," + ",".join(prop_keys) + "\n"
    for smiles in smiles_keys:
        vals = [str(dic[smiles][prop_key]) for prop_key in prop_keys]
        text += ",".join([smiles] + vals) + "\n"

    with open(path, "w") as f:
        f.write(text)


def collect_csvs(prop_name,
                 resave,
                 base_save_path=BASE_SAVE_PATH):

    combined_csv_name = "{}_combined.csv".format(prop_name)
    combined_path = os.path.join(base_save_path, combined_csv_name)

    if os.path.isfile(combined_path) and not resave:
        return combined_path

    csv_names = []
    re_str = "{}_\\d+.csv".format(prop_name)
    for file in os.listdir(base_save_path):
        csv_names += re.findall(re_str, file)

    overall_dict = {}
    for csv_name in csv_names:
        path = os.path.join(base_save_path, csv_name)
        this_dic, _ = read_csv(path)
        overall_dict.update(this_dic)

    write_csv(combined_path, overall_dict)

    return combined_path


def collect_features(feat_name,
                     prop_name,
                     prop_csv_path,
                     resave,
                     base_save_path=BASE_SAVE_PATH):

    combined_feat_name = "{}_{}_combined.npz".format(feat_name, prop_name)
    combined_path = os.path.join(base_save_path, combined_feat_name)

    if os.path.isfile(combined_path) and not resave:
        return combined_path

    file_names = []
    re_str = "{}_\\d+.npz".format(feat_name)
    for file in os.listdir(base_save_path):
        file_names += re.findall(re_str, file)

    overall_dict = {}

    for file in file_names:
        path = os.path.join(base_save_path, file)
        filedate = datetime.utcfromtimestamp(os.path.getmtime(path))
        delta = filedate - datetime.now()
        seconds = abs((delta.days * 24 * 60 * 60 + delta.seconds))

        if seconds < 60:
            continue
        data = np.load(os.path.join(base_save_path, file))

        for smiles, feats in zip(data["smiles"], data["features"]):
            overall_dict[smiles] = feats

    ordered_smiles, _ = read_csv(prop_csv_path)
    ordered_feats = np.array([overall_dict[smiles]
                              for smiles in ordered_smiles])

    np.savez_compressed(
        combined_path, features=ordered_feats, smiles=ordered_smiles)

    return combined_path


def run_chemprop(csv_path,
                 features_path,
                 save_dir,
                 features_only,
                 dropout,
                 base_save_path=BASE_SAVE_PATH,
                 device=0):

    cmd = ("python $HOME/Repo/projects/chemprop/train.py --data_path {0}"
           " --dataset_type regression --save_dir {1}"
           " --save_smiles_splits "
           " --no_features_scaling --quiet  --gpu {2} --num_folds 1 "
           " --metric 'mae' --dropout {3} ").format(
        csv_path, save_dir,
        device, dropout)

    if features_path is not None:
        cmd += " --features_path {}".format(features_path)

    if features_only:
        cmd += " --features_only"

    cmds = ["source deactivate", "source activate chemprop",
            cmd]

    for cmd in cmds:
        try:
            subprocess.check_output([cmd], shell=True).decode()
        except Exception as e:
            print(e)
            continue


def get_best_val_score(save_dir):

    log_path = os.path.join(save_dir, "quiet.log")
    with open(log_path, "r") as f:
        lines = f.readlines()
        # must reverse because different training logs
        # get appended to the same file
        for line in reversed(lines):
            if "Overall test" in line:
                score = float(line.split("=")[1].split()[0])
                return score


def evaluate_model(prop_name,
                   feat_name,
                   features_only,
                   save_dir,
                   dropout,
                   resave_feats=False,
                   resave_csv=False,
                   base_save_path=BASE_SAVE_PATH,
                   device=0):

    csv_path = collect_csvs(prop_name=prop_name,
                            resave=resave_csv,
                            base_save_path=base_save_path)

    if feat_name is not None:
        feat_path = collect_features(feat_name=feat_name,
                                     prop_name=prop_name,
                                     prop_csv_path=csv_path,
                                     resave=resave_feats,
                                     base_save_path=base_save_path)
    else:
        feat_path = None

    run_chemprop(csv_path=csv_path,
                 features_path=feat_path,
                 save_dir=save_dir,
                 features_only=features_only,
                 base_save_path=base_save_path,
                 device=device,
                 dropout=dropout)

    score = get_best_val_score(save_dir)

    return score


def run_expt(conn, experiment, **kwargs):
    i = 0
    while (experiment.progress.observation_count
            < experiment.observation_budget):

        suggestion = conn.experiments(experiment.id
                                      ).suggestions().create()
        dropout = np.exp(suggestion.assignments["log_dropout"])
        if i > 0:
            kwargs.update({"resave_feats": False,
                           "resave_csv": False})
        try:
            value = evaluate_model(dropout=dropout, **kwargs)
        except Exception as e:
            print(e)
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                failed=True)

            continue

        print(value)

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        experiment = conn.experiments(experiment.id).fetch()

        i += 1


def main(feats=FEATS,
         props=PROPS,
         base_save_path=BASE_SAVE_PATH,
         base_chemprop_path=BASE_CHEMPROP_PATH,
         device=0,
         token=TOKEN,
         resave_feats=True,
         resave_csv=True):

    procs = []
    all_feats = feats + [None]

    for i, feat_name in enumerate(all_feats):
        for j, prop_name in enumerate(props):
            for features_only in [True, False]:

                iter_name = "{}_{}_mpnn_{}".format(prop_name,
                                                   feat_name,
                                                   json.dumps(
                                                       (not features_only)))
                if feat_name is None:
                    if features_only:
                        continue
                    else:
                        iter_name = "{}_only_mpnn".format(prop_name)

                save_dir = os.path.join(base_chemprop_path, iter_name)
                conn, experiment = make_expt(name=iter_name, token=token)

                kwargs = dict(feat_name=feat_name,
                              prop_name=prop_name,
                              resave_feats=resave_feats,
                              resave_csv=resave_csv,
                              base_save_path=base_save_path,
                              device=device,
                              save_dir=save_dir,
                              features_only=features_only)

                p = Process(target=run_expt, args=(conn, experiment),
                            kwargs=kwargs)
                p.start()
                procs.append(p)

    for p in procs:
        p.join()

# NEED TO USE THE SAME TRAIN  / VAL  / TEST SPLITS!!!!!!
# AND ONLY TRAIN ON A SUBSET OF THE DATA!!


if __name__ == "__main__":

    main()
