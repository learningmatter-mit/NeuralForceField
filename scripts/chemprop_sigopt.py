from sigopt import Connection
import re
import os
import numpy as np
import subprocess
import json
from threading import Thread

TOKEN = "KTNMWLZQYQSNCHVHPGIWSAVXEWLEWABZAHIJOLXKWAHQDRQE"
BASE_SAVE_PATH = ("/pool001/saxelrod/data_from_fock/"
                  "combined_fingerprint_datasets")
BASE_CHEMPROP_PATH = "/home/saxelrod/chemprop_sigopt"
FEATS = ['mean_e3fp', 'morgan']
PROPS = ["ensembleentropy"]


def make_expt(name, token=TOKEN):
    conn = Connection(client_token=token)
    experiment = conn.experiments().create(
        name=name,
        metrics=[dict(name='mae', objective='minimize')],
        parameters=[
            dict(name='log_dropout', type='float', bounds=dict(min=-5, max=0)),
        ],
        observation_budget=20
    )

    return conn, experiment


def read_csv(path):

    dic = {}
    ordered_smiles = []

    with open(path, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            smiles = line.split(",")[0]
            prop = line.split(",")[1]
            dic[smiles] = prop
            ordered_smiles.append(smiles)

    return dic, ordered_smiles


def write_csv(path, dic):

    smiles_keys = sorted(list(dic.keys()))
    sub_dic_0 = dic[smiles_keys[0]]
    prop_keys = sorted(list(sub_dic_0.keys()))

    text = "smiles," + ",".join(prop_keys) + "\n"
    for smiles in smiles_keys:
        vals = [dic[smiles][prop_key] for prop_key in prop_keys]
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
    re_str = "{}_\\d.csv".format(prop_name)
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

    combined_feat_name = "{}_{}.npz".format(feat_name, prop_name)
    combined_path = os.path.join(base_save_path, combined_feat_name)

    if os.path.isfile(combined_path) and not resave:
        return combined_path

    file_names = []
    re_str = "{}_\\d.csv".format(feat_name)
    for file in os.listdir(base_save_path):
        file_names += re.findall(re_str, file)

    overall_dict = {}

    for file in file_names:

        data = np.load(file)
        overall_dict[data["smiles"]] = data["feats"]

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

    cmd = ("python train.py --data_path {0}"
           " --dataset_type regression --save_dir {1}"
           " --save_smiles_splits  --features_path {2} "
           " --no_features_scaling --quiet  --gpu {3} --num_folds 1 "
           " --metric 'mae' --dropout {4} ").format(
        csv_path, save_dir,
        features_path, device, dropout)

    if features_only:
        cmd += " --features_only"

    cmds = ["conda deactivate", "conda activate chemprop", cmd]
    outputs = []

    for cmd in cmds:
        outputs.append(subprocess.check_output([cmd], shell=True).decode())

    return outputs


def get_best_val_score(text):
    for line in reversed(text.split("\n")):
        if "best validation" in line:
            score = float(line.split("=")[1].split()[0])
            return score


def evaluate_model(prop_name,
                   feat_name,
                   features_only,
                   save_dir,
                   resave_feats=False,
                   resave_csv=False,
                   base_save_path=BASE_SAVE_PATH,
                   device=0):

    csv_path = collect_csvs(prop_name=prop_name,
                            resave=resave_csv,
                            base_save_path=base_save_path)

    feat_path = collect_features(feat_name=feat_name,
                                 prop_name=prop_name,
                                 prop_csv_path=csv_path,
                                 resave=resave_feats,
                                 base_save_path=base_save_path)

    outputs = run_chemprop(csv_path=csv_path,
                           features_path=feat_path,
                           save_dir=save_dir,
                           features_only=features_only,
                           base_save_path=base_save_path,
                           device=device)

    score = get_best_val_score(outputs[-1])

    return score


def run_expt(conn, experiment, **kwargs):

    while (experiment.progress.observation_count
           < experiment.observation_budget):

        suggestion = conn.experiments(experiment.id).suggestions().create()
        dropout = np.exp(suggestion.assignments["log_dropout"])
        value = evaluate_model(dropout=dropout, **kwargs)
        print(value)

        conn.experiments(experiment.id).observations().create(
            suggestion=suggestion.id,
            value=value,
        )

        experiment = conn.experiments(experiment.id).fetch()


def main(feats=FEATS,
         props=PROPS,
         resave_feats=False,
         resave_csv=False,
         base_save_path=BASE_SAVE_PATH,
         base_chemprop_path=BASE_CHEMPROP_PATH,
         device=0,
         token=TOKEN):

    threads = []
    for feat_name in feats:
        for prop_name in props:
            for features_only in [True, False]:

                iter_name = "{}_{}_mpnn_{}".format(prop_name,
                                                   feat_name,
                                                   json.dumps(
                                                       (not features_only)))

                save_dir = os.path.join(base_chemprop_path, iter_name)
                conn, experiment = make_expt(name=iter_name, token=token)
                thread = Thread(target=run_expt(conn,
                                                experiment,
                                                feat_name=feat_name,
                                                prop_name=prop_name,
                                                resave_feats=resave_feats,
                                                resave_csv=resave_csv,
                                                base_save_path=base_save_path,
                                                device=device,
                                                save_dir=save_dir))

                thread.start()
                threads.append(thread)

    for thread in threads:
        thread.join()


