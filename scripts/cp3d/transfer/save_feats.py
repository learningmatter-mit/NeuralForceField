import os
import pickle
import numpy as np
import argparse

PATH = "/home/saxelrod/supercloud2/models/cov_2_protease"
# MODEL_NAME = "attention_k_1_no_prob_cov_cl_protease"
MODEL_NAME = "attention_k_3_yes_prob_cov_cl_protease"
SAVE_DIR = "/home/saxelrod/chemprop_cov_2/features"


def get_smiles(name):
    with open(name, "r") as f:
        lines = f.readlines()
    smiles_list = [i.strip() for i in lines[1:]]
    return smiles_list


def save_smiles(smiles_list, name):
    paths = [f"{name}_smiles.csv", f"{name}_full.csv"]
    for path in paths:
        with open(path, "r") as f:
            lines = f.readlines()
        keep_lines = [lines[0]]
        for line in lines[1:]:
            smiles = line.split(",")[0].strip()
            if smiles in smiles_list:
                keep_lines.append(line)
        text = "".join(keep_lines)
        with open(path, "w") as f:
            f.write(text)


def main(path,
         model_name,
         save_dir):
    metrics = ["loss", "roc_auc", "prc_auc"]
    for metric in metrics:
        names = ["train", "val", "test"]
        save_path = os.path.join(path, model_name)
        for name in names:
            file_name = f"pred_{metric}_{name}.pickle"
            file_path = os.path.join(save_path, file_name)
            with open(file_path, "rb") as f:
                dic = pickle.load(f)

            smiles_list = get_smiles(f"{name}_smiles.csv")
            smiles_list = [smiles for smiles in smiles_list if smiles in dic]
            save_smiles(smiles_list, name)

            ordered_feats = np.stack([dic[smiles]["fp"]
                                      for smiles in smiles_list])
            np_save_path = os.path.join(save_dir, model_name,
                                        f"{name}_{metric}.npz")

            np.savez(np_save_path, features=ordered_feats)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        help=("Path to fingerprint pickles"),
                        default=PATH)
    parser.add_argument('--model_name', type=str,
                        help=("Name of model used for fingerprints"),
                        default=MODEL_NAME)
    parser.add_argument('--save_dir', type=str,
                        help=("Where to save the features"),
                        default=SAVE_DIR)
    args = parser.parse_args()

    main(path=args.path,
         model_name=args.model_name,
         save_dir=args.save_dir)
