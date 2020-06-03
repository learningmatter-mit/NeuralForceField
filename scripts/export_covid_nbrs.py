import msgpack
import pickle
import os

BASE_PATH = "/pool001/saxerod/data_from_fock/final_db_data"
SAVE_FOLDER = "/home/saxelrod/fock/chemdata/notebooks/saxelrod"


def to_list_of_dics(out):

    for key, sub_dic in out.items():
        list_of_dics = []
        sub_dic_keys = list(sub_dic.keys())
        num_confs = len(sub_dic[sub_dic_keys[0]])
        for sub_key in sub_dic_keys:
            assert num_confs == len(sub_dic[sub_key])

        for i in range(num_confs):
            new_dic = {key: sub_dic[key][i] for key in sub_dic_keys}
            list_of_dics.append(new_dic)

        out[key] = list_of_dics

    return out


def mp_save(group, nbrs, save_folder=SAVE_FOLDER, stop=None, remove_old=False,
            base_path=BASE_PATH):

    i = 0

    if group == 'qm9':
        group_alias = 'qm9'
    else:
        group_alias = 'drugs'
    if nbrs:
        save_name = "{}_post_process.msgpack".format(group_alias)
    else:
        save_name = "{}_ensembles.msgpack".format(group_alias)

    save_path = os.path.join(save_folder, save_name)
    if os.path.isfile(save_path) and remove_old:
        os.remove(save_path)

    for file in os.listdir(base_path):

        condition = file.startswith(group
                                    ) and file.endswith("pickle"
                                                        ) and "old" not in file
        if nbrs:
            condition = condition and "nbrs" in file
        else:
            condition = condition and "nbrs" not in file
        if condition:

            print(i)
            path = os.path.join(base_path, file)
            with open(path, "rb") as f:
                out = pickle.load(f)

            if nbrs:
                for key, sub_dic in out.items():
                    sub_dic.pop("rd_mols")
                list_of_dics = to_list_of_dics(out)
                mp_form = msgpack.packb(list_of_dics, use_bin_type=True)
            else:
                mp_form = msgpack.packb(out, use_bin_type=True)

            with open(save_path, "ab") as f:
                f.write(mp_form)

            i += 1

        if stop is not None and i >= stop:
            break


if __name__ == "__main__":
    mp_save(group='covid', nbrs=True)
