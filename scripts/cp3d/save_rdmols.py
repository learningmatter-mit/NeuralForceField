from rdkit import Chem
import os
import pickle
import msgpack
import json

base_dir = "/home/saxelrod/engaging_nfs/data_from_fock/final_db_data"
geom_dir = "/home/saxelrod/Repo/projects/geom/tutorials"

nobackup = '/home/saxelrod/nobackup/data'
rd_drugs_file = os.path.join(nobackup, "drugs_rd_mols.pickle")
rd_qm9_file = os.path.join(nobackup, "qm9_rd_mols.pickle")

data_path = '/home/saxelrod/engaging_nfs/data_from_fock/final_db_data'
drugs_file = os.path.join(geom_dir, "drugs_crude.msgpack")
qm9_file = os.path.join(geom_dir, "qm9_crude.msgpack")

drugs_save_folder = ("/home/saxelrod/Repo/projects/geom/tutorials/"
                     "rd_folder/drugs")
qm9_save_folder = "/home/saxelrod/Repo/projects/geom/tutorials/rd_folder/qm9"

drugs_summary_path = ("/home/saxelrod/Repo/projects/geom/"
                      "tutorials/rd_folder/summary_drugs_all.json")
qm9_summary_path = ("/home/saxelrod/Repo/projects/geom/"
                    "tutorials/rd_folder/summary_qm9_all.json")


def get_smiles(mol):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
    return smiles


def add_rd_mols(remove_old=True, start_idx=0, crude_file=qm9_file,
                rd_file=rd_qm9_file):

    print("Iterating through crude file...")
    unpacker = msgpack.Unpacker(open(crude_file, "rb"))

    if os.path.isfile(rd_file) and remove_old:
        os.remove(rd_file)

    print("Loading crude file...")
    for i, dic in enumerate(unpacker):
        print(i)
        if i < start_idx:
            continue
        pickle_path = os.path.join(
            data_path, "covid_fock_{}_nbrs.pickle".format(i+1))
        print("Loading pickle...")
        try:
            with open(pickle_path, "rb") as f:
                rd_data = pickle.load(f)
        except Exception as e:
            print(e)
            continue
        new_dic = {}
        for key, sub_dic in dic.items():
            if key not in rd_data:
                continue
            new_dic[key] = {sub_key: sub_val for sub_key, sub_val in
                            sub_dic.items()}

            rd_mols = rd_data[key]["rd_mols"]
            confs = new_dic[key]["conformers"]
            assert len(confs) == len(rd_mols)

            for i, rd_mol in enumerate(rd_mols):
                confs[i].pop("xyz")
                confs[i]["rd_mol"] = rd_mol

        print("{} entries in new dictionary".format(len(new_dic)))
        print("Loading crude file...")

        with open(rd_file, "ab") as f:
            pickle.dump(new_dic, f)


def rename_save(save_path):
    j = 1
    while os.path.isfile(save_path):
        save_path = save_path.replace(".pickle", "") + "_{}".format(j)
        save_path += ".pickle"
        j += 1
    return save_path


def save_dic(dic, save_folder):

    file_dic = {}

    for smiles, sub_dic in dic.items():
        save_name = smiles.replace("/", "_") + ".pickle"
        save_path = os.path.join(save_folder, save_name)
        sub_dic["smiles"] = smiles

#         save_path = rename_save(save_path)

        while True:
            try:
                #                 save_path = rename_save(save_path)
                with open(save_path, "wb") as f:
                    pickle.dump(sub_dic, f)
                file_dic[smiles] = save_path
                break
            except OSError as e:
                print(e)
                while True:
                    save_path = save_path.replace(".pickle", "")[:-1]
                    save_path += ".pickle"

                    if not os.path.isfile(save_path):
                        break

    return file_dic


def count_pickle_path(dic, name):
    num = len(([i['pickle_path'] for i in dic.values() if 'pickle_path' in i]))
    print("Number of SMILES in {} with pickle paths is {}".format(name, num))


def main(save_folder=qm9_save_folder, rd_file=rd_qm9_file,
         summary_path=qm9_summary_path):

    file_dic = {}

    f = open(rd_file, "rb")
    smiles_list = []

    for i in range(300):

        try:
            dic = pickle.load(f)
        except Exception as e:
            print(e)
            break

        new_file_dic = save_dic(dic, save_folder)
        file_dic.update(new_file_dic)

        smiles_list += list(dic.keys())
        num_save = len(list(set(smiles_list)))

        print("Saved {} mols".format(num_save))
        print("Length of file dic is {}".format(
            len(list(set(file_dic.values())))))

    f.close()

    # add to dictionary

    with open(summary_path, "r") as f:
        test_summary = json.load(f)

    print("Number of SMILES in summary dic is {}".format(len(
        test_summary)))

    for key, val in test_summary.items():
        if key in file_dic:
            file_val = "/".join(file_dic[key].split("/")[-2:])
            # print(file_val)
            val["pickle_path"] = file_val

    count_pickle_path(test_summary, "saved dictionary")

    with open(summary_path, "w") as f:
        json.dump(test_summary, f, indent=4, sort_keys=True)

    with open(summary_path, "r") as f:
        test_summary = json.load(f)

    count_pickle_path(test_summary, "loaded dictionary")


if __name__ == "__main__":
    main()
