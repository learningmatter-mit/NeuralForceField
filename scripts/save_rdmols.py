import sys
sys.path.insert(0, "/home/saxelrod/Repo/projects/covid_nff/NeuralForceField")


from nff.data.features.graph import (bond_feats_from_dic, atom_feats_from_dic,
    nbr_list_from_dic)
from rdkit import Chem
import os
import pickle
# import msgpack
import json
import re
import copy


# base_dir = "/home/saxelrod/engaging_nfs/data_from_fock/final_db_data"
# geom_dir = "/home/saxelrod/Repo/projects/geom/tutorials"

# nobackup = '/home/saxelrod/nobackup/data'
# rd_drugs_file = os.path.join(nobackup, "drugs_rd_mols.pickle")
# rd_qm9_file = os.path.join(nobackup, "qm9_rd_mols.pickle")

# data_path = '/home/saxelrod/engaging_nfs/data_from_fock/final_db_data'
# drugs_file = os.path.join(geom_dir, "drugs_crude.msgpack")
# qm9_file = os.path.join(geom_dir, "qm9_crude.msgpack")

# drugs_save_folder = ("/home/saxelrod/Repo/projects/geom/tutorials/"
#                      "rd_folder/drugs")
# qm9_save_folder = "/home/saxelrod/Repo/projects/geom/tutorials/rd_folder/qm9"

# drugs_summary_path = ("/home/saxelrod/Repo/projects/geom/"
#                       "tutorials/rd_folder/summary_drugs_all.json")
# qm9_summary_path = ("/home/saxelrod/Repo/projects/geom/"
#                     "tutorials/rd_folder/summary_qm9_all.json")


def get_smiles(mol):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
    return smiles


def featurize_sub_dic(sub_dic):

    dic_list = sub_dic["rd_mols"]

    bond_feats = bond_feats_from_dic(dic_list)
    atom_feats = atom_feats_from_dic(dic_list)
    bonded_nbr_list = nbr_list_from_dic(dic_list)

    sub_dic.update({"bond_features": bond_feats,
                    "atom_features": atom_feats,
                    "bonded_nbr_list": bonded_nbr_list})

    return sub_dic


# class PickleIter(list):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.counter = 0

#     def __iter__(self):
#         self.counter = 0
#         return self

#     def __next__(self):

#         if self.counter == len(self):
#             raise StopIteration

#         item = self.__getitem__(self.counter)
#         with open(item, "rb") as f:
#             dic = pickle.load(f)
#         self.counter += 1

#         return dic


def get_pattern(project, use_msgpack, nbrs):
    if use_msgpack:
        pattern = "{}_crude.msgpack".format(project)
    else:
        pattern = r"{}_fock_\d+.pickle".format(project)
        if nbrs:
            pattern = pattern.replace(".pickle", "_nbrs.pickle")

    return pattern


# def get_unpacker(direc, project, use_msgpack, nbrs):

#     pattern = get_pattern(project, use_msgpack, nbrs)

#     if use_msgpack:
#         path = os.path.join(direc, pattern)
#         unpacker = msgpack.Unpacker(open(path, "rb"))
#     else:

#         paths = sorted([os.path.join(direc, i) for i in os.listdir(direc)
#                         if re.findall(pattern, i)])
#         unpacker = PickleIter(paths)

#     return unpacker


def make_pickle_dic(file_dic,
                    nbr_dic,
                    smiles,
                    confs):

    pickle_dic = copy.deepcopy(file_dic[smiles])
    rd_mols = nbr_dic[smiles]["rd_mols"]

    assert len(rd_mols) == len(confs)

    for i, conf in enumerate(confs):
        rd_mol = rd_mols[i]
        confs[i]["rd_mol"] = rd_mol
        confs[i].pop("xyz")

    pickle_dic["conformers"] = confs
    pickle_dic["smiles"] = smiles

    return pickle_dic


def save_pickle(pickle_dic,
                smiles,
                save_folder,
                project):

    save_name = smiles.replace("/", "_") + ".pickle"
    proj_folder = os.path.join(save_folder, project)

    if not os.path.isdir(proj_folder):
        os.makedirs(proj_folder)

    while True:
        try:
            path = os.path.join(proj_folder, save_name)
            if os.path.isfile(path):
                raise OSError("Path exists")
            with open(path, "wb") as f:
                pickle.dump(pickle_dic, f)

            break
        except OSError as e:
            print(e)
            save_name = save_name.replace(".pickle", "")[:-1]

    pickle_path = os.path.join(project, save_name)
    return pickle_path


def save_rdmols_as_singles(direc,
                           project,
                           save_folder):

    file_dic = {}

    # crude_unpacker = get_unpacker(direc=direc,
    #                               project=project,
    #                               use_msgpack=False,
    #                               nbrs=False)

    # nbr_unpacker = get_unpacker(direc=direc,
    #                             project=project,
    #                             use_msgpack=False,
    #                             nbrs=True)

    file_dic = {}
    saved_pickles = 0
    # for crude, nbr in zip(crude_unpacker, nbr_unpacker):

    pattern = get_pattern(project, use_msgpack=False, nbrs=False)
    crude_files = [os.path.join(direc, i) for i in os.listdir(direc)
                   if re.findall(pattern, i)]

    for crude_file in crude_files:

        nbr_file = crude_file.replace(".pickle", "_nbrs.pickle")
        # print("Analyzing crude file {} and nbr_file {}".format(
        #     crude_file, nbr_file))

        ####
        if not os.path.isfile(nbr_file):
            continue
        ####

        with open(crude_file, "rb") as f:
            crude = pickle.load(f)
        with open(nbr_file, "rb") as f:
            nbr = pickle.load(f)

        for smiles, crude_dic in crude.items():
            file_dic[smiles] = {sub_key: sub_val for
                                sub_key, sub_val in crude_dic.items()
                                if "conformers" not in sub_key}

            if smiles in nbr:

                confs = crude_dic["conformers"]
                pickle_dic = make_pickle_dic(file_dic=file_dic,
                                             nbr_dic=nbr,
                                             smiles=smiles,
                                             confs=confs)

                pickle_path = save_pickle(pickle_dic=pickle_dic,
                                          smiles=smiles,
                                          save_folder=save_folder,
                                          project=project)

                file_dic["pickle_path"] = pickle_path
                saved_pickles += 1
        print("Saved {} pickle files".format(saved_pickles))
        print("{} smiles in file_dic".format(len(file_dic)))

        #####
        break
        #####

    print("Saving summary file...")
    file_dic_path = os.path.join(direc, "summary_{}.json".format(project))
    with open(file_dic_path, "w") as f:
        json.dump(file_dic, f, indent=4, sort_keys=True)
    print("Saved. Done!")

    # for file in os.listdir(nobackup_folder):
    #     feat_path = os.path.join(nobackup_folder, file)
    #     with open(feat_path, "rb") as f:
    #         out = pickle.load(f)
    #     for key, val in out.items():

    #         bonded_nbr_tuples = val['bonded_nbr_tuples']
    #         if len(bonded_nbr_tuples) != 1:
    #             print(key)
    #             continue

    #         save_name = key.replace("/", "_") + ".pickle"
    #         dic = {key: val[key] for key in ['atom_features', 'bond_features']}
    #         dic.update(
    #             {"smiles": key, 'bonded_nbr_list': val['bonded_nbr_tuples'][0][0]})

    #         while True:
    #             try:
    #                 path = os.path.join(save_folder, save_name)
    #                 if os.path.isfile(path):
    #                     raise OSError("Path exists")
    #                 with open(path, "wb") as f:
    #                     pickle.dump(dic, f)
    #                 file_dic[key] = path
    #                 break
    #             except OSError as e:
    #                 print(e)
    #                 save_name = save_name.replace(".pickle", "")[:-1]
    #                 path = os.path.join(save_folder, save_name)

    #     unique_smiles += list(out.keys())
    #     unique_smiles = list(set(unique_smiles))
    #     print("Saved {} smiles".format(len(unique_smiles)))

    # with open(file_dic_path, "w") as f:
    #     json.dump(file_dic, f, indent=4, sort_keys=True)


# def add_rd_mols(crude_direc,
#                 use_msgpack,
#                 # save_separate,
#                 remove_old=True,
#                 start_idx=0,
#                 rd_file=rd_qm9_file):

#     print("Iterating through crude file...")

#     unpacker = get_unpacker(crude_direc, use_msgpack)

#     if os.path.isfile(rd_file) and remove_old:
#         os.remove(rd_file)

#     print("Loading crude file...")
#     i = 0

#     for i, dic in enumerate(unpacker):
#         print(i)
#         if i < start_idx:
#             continue
#         pickle_path = os.path.join(
#             data_path, "covid_fock_{}_nbrs.pickle".format(i+1))
#         print("Loading pickle...")
#         try:
#             with open(pickle_path, "rb") as f:
#                 rd_data = pickle.load(f)
#         except Exception as e:
#             print(e)
#             continue
#         new_dic = {}
#         for key, sub_dic in dic.items():
#             if key not in rd_data:
#                 continue
#             new_dic[key] = {sub_key: sub_val for sub_key, sub_val in
#                             sub_dic.items()}

#             rd_mols = rd_data[key]["rd_mols"]
#             confs = new_dic[key]["conformers"]
#             assert len(confs) == len(rd_mols)

#             for i, rd_mol in enumerate(rd_mols):
#                 confs[i].pop("xyz")
#                 confs[i]["rd_mol"] = rd_mol

#         print("{} entries in new dictionary".format(len(new_dic)))
#         print("Loading crude file...")

#         # so this is just appending all the pickles to one
#         # big pickle file. Nothing more. Not saving them separately

#         with open(rd_file, "ab") as f:
#             pickle.dump(new_dic, f)


# def rename_save(save_path):
#     j = 1
#     while os.path.isfile(save_path):
#         save_path = save_path.replace(".pickle", "") + "_{}".format(j)
#         save_path += ".pickle"
#         j += 1
#     return save_path


# def save_dic(dic, save_folder):

#     file_dic = {}

#     for smiles, sub_dic in dic.items():
#         save_name = smiles.replace("/", "_") + ".pickle"
#         save_path = os.path.join(save_folder, save_name)
#         sub_dic["smiles"] = smiles

# #         save_path = rename_save(save_path)

#         while True:
#             try:
#                 #                 save_path = rename_save(save_path)
#                 with open(save_path, "wb") as f:
#                     pickle.dump(sub_dic, f)
#                 file_dic[smiles] = save_path
#                 break
#             except OSError as e:
#                 print(e)
#                 while True:
#                     save_path = save_path.replace(".pickle", "")[:-1]
#                     save_path += ".pickle"

#                     if not os.path.isfile(save_path):
#                         break

#     return file_dic


# def count_pickle_path(dic, name):
#     num = len(([i['pickle_path'] for i in dic.values() if 'pickle_path' in i]))
#     print("Number of SMILES in {} with pickle paths is {}".format(name, num))


# def main(save_folder=qm9_save_folder, rd_file=rd_qm9_file,
#          summary_path=qm9_summary_path):

#     file_dic = {}

#     f = open(rd_file, "rb")
#     smiles_list = []

#     for i in range(300):

#         try:
#             dic = pickle.load(f)
#         except Exception as e:
#             print(e)
#             break

#         new_file_dic = save_dic(dic, save_folder)
#         file_dic.update(new_file_dic)

#         smiles_list += list(dic.keys())
#         num_save = len(list(set(smiles_list)))

#         print("Saved {} mols".format(num_save))
#         print("Length of file dic is {}".format(
#             len(list(set(file_dic.values())))))

#     f.close()

#     # add to dictionary

#     with open(summary_path, "r") as f:
#         test_summary = json.load(f)

#     print("Number of SMILES in summary dic is {}".format(len(
#         test_summary)))

#     for key, val in test_summary.items():
#         if key in file_dic:
#             file_val = "/".join(file_dic[key].split("/")[-2:])
#             # print(file_val)
#             val["pickle_path"] = file_val

#     count_pickle_path(test_summary, "saved dictionary")

#     with open(summary_path, "w") as f:
#         json.dump(test_summary, f, indent=4, sort_keys=True)

#     with open(summary_path, "r") as f:
#         test_summary = json.load(f)

#     count_pickle_path(test_summary, "loaded dictionary")


if __name__ == "__main__":
    # main()
    save_folder = "/home/saxelrod/rgb_nfs/GEOM_DATA_ROUND_2/rdkit_folder"
    project = "covid"
    direc = "/home/saxelrod/engaging_nfs/data_from_fock/final_db_data"

    save_rdmols_as_singles(direc=direc,
                           project=project,
                           save_folder=save_folder)
