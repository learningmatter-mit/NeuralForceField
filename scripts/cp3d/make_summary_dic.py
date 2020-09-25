import pickle
import numpy as np
import copy
import json
import os
import re

BASE_PATH = "/pool001/saxelrod/data_from_fock/final_db_data"
RD_PATH = "/nfs/rafagblab001/saxelrod/GEOM_DATA_ROUND_2/rdkit_folder/covid"
PATTERN = r"covid_fock_\d+.pickle"
SAVE_PATH = ("/nfs/rafagblab001/saxelrod/GEOM_DATA_ROUND_2"
             "/rdkit_folder/covid/summary_dic.json")

NUM_SPLITS = 5

TEST_SMILES = ('CC[C@@H]1[C@@H](C)O[C@@](O)([C@@H](C)[C@H](O)'
               '[C@H](C)[C@H]2OC(=O)/C=C/C=C/[C@H](C)[C@H]([C@@H](C)[C@@H]'
               '(O)[C@H](C)[C@@]3(O)C[C@@H](O[C@H]4C[C@H](O)[C@H](O)[C@H](C)O4)'
               '[C@H](CC)[C@@H](C)O3)OC(=O)/C=C/C=C/[C@H]2C)C[C@H]1O[C@H]1C[C@H]'
               '(O)[C@H](O)[C@H](C)O1')


def add_pickle_ext(rd_path, file):
    new_name = file
    new_path = os.path.join(rd_path, new_name + ".pickle")
    original_path = os.path.join(rd_path, new_name)
    with open(original_path, "rb") as f:
        pickle_dic = pickle.load(f)
    while True:
        try:
            with open(new_path, "wb") as f:
                pickle.dump(pickle_dic, f)
            break
        except OSError as e:
            print(e)
            new_name = new_name[:-1]
            new_path = os.path.join(rd_path, new_name + ".pickle")
    os.remove(original_path)
    return new_name + ".pickle"


def get_rd_file(smiles, all_files, rd_path):
    if smiles in all_files:
        return all_files[smiles]

    base_smiles = smiles.replace("/", "_")
    arr = np.array(list(base_smiles))
    splits = ["".join(i) for i in np.array_split(arr, NUM_SPLITS)]
    start_split = splits[0]
    possible_files = list(all_files.values())

    for i in range(NUM_SPLITS):
        possible_files = list(filter(
            lambda x: x.startswith(start_split), possible_files))

        if len(possible_files) == 1:
            real_file = possible_files[0]
            if not real_file.endswith(".pickle"):
                real_file = add_pickle_ext(rd_path, real_file)
            return real_file

        if i == NUM_SPLITS - 1:
            break
        start_split += splits[i + 1]


def main(base_path=BASE_PATH,
         pattern=PATTERN,
         rd_path=RD_PATH,
         save_path=SAVE_PATH):
    paths = [os.path.join(base_path, file) for file in
             os.listdir(base_path) if re.findall(pattern, file)]
    summary_dic = {}
    all_files = {i.replace("_", "/").replace(".pickle", "")
                           : i for i in os.listdir(rd_path)}
    length = len(paths)

    for i, path in enumerate(paths):
        print("Starting path {} of {}".format(i, length))
        with open(path, "rb") as f:
            dic = pickle.load(f)
        num_rd = 0
        actual = 0
        for smiles, sub_dic in dic.items():
            new_dic = {sub_key: sub_val for sub_key, sub_val
                       in sub_dic.items() if sub_key != "conformers"}
            rd_file = get_rd_file(smiles, all_files, rd_path)
            if rd_file is not None:
                new_dic["pickle_path"] = "drugs/{}".format(rd_file)
                num_rd += 1
                print("Got rd_file {}".format(num_rd))
                actual += 1
            summary_dic.update({smiles: new_dic})

        possible = len(dic)

        print("Got pickles for {} of {}".format(actual, possible))
        print("Finished {}".format(i))

    with open(save_path, "w") as f:
        json.dump(summary_dic, f)


if __name__ == "__main__":
    main()
