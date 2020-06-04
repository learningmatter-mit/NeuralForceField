from nff.data import Dataset, concatenate_dict
import pdb
import argparse
import numpy as np
import torch
import msgpack
import sys
sys.path.insert(0, "/home/saxelrod/repo/nff/covid/NeuralForceField")


KEY_MAP = {"xyz": "nxyz", "boltzmannweight": "weights",
           "relativeenergy": "energy"}

EXCLUDE_KEYS = ["totalconfs", "datasets"]
MAX_ATOMS = 60


def load_data(file, num_specs, max_atoms, smiles_csv=None):

    unpacker = msgpack.Unpacker(open(file, "rb"))
    spec_count = 0
    overall_dic = {}

    smiles_dic = None
    if smiles_csv is not None:
        smiles_dic = {}
        with open(smiles_csv, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines[1:]):
            smiles = line.split(",")[0]
            smiles_dic[smiles] = i        

    for chunk in unpacker:
        for key, val in chunk.items():

            val = chunk[key]
            num_atoms = len(val["conformers"][0]["xyz"])
            this_smiles = key

            if num_atoms > max_atoms:
                continue
            if smiles_dic is not None and not smiles_dic.get(this_smiles):
                print("Skipping smiles {}".format(smiles))
                continue

            overall_dic.update({key: val})
            spec_count += 1

            if spec_count >= num_specs:
                print("Loaded {} species".format(spec_count))
                return overall_dic

        print("Loaded {} species".format(spec_count))

    return overall_dic


def map_key(key):
    if key in KEY_MAP:
        return KEY_MAP[key]
    else:
        return key


def fix_iters(spec_dic, actual_confs):
    new_spec_dic = {}
    for key, val in spec_dic.items():
        if key in EXCLUDE_KEYS:
            continue
        if type(val) in [int, float]:
            new_spec_dic[key] = [val] * actual_confs

    return new_spec_dic


def convert_data(overall_dic, num_confs):

    spec_dics = []
    for key, sub_dic in overall_dic.items():
        spec_dic = {map_key(key): val for key, val in sub_dic.items()
                    if key != "conformers"}
        if spec_dic["charge"] != 0:
            continue
        actual_confs = min(num_confs, len(sub_dic["conformers"]))
        spec_dic = fix_iters(spec_dic, actual_confs)
        spec_dic.update({map_key(key): [] for key
                         in sub_dic["conformers"][0].keys()})

        for conf in sub_dic["conformers"][:num_confs]:
            for key in conf.keys():
                spec_dic[map_key(key)].append(conf[key])
        spec_dics.append(spec_dic)

    return spec_dics


def make_nff_dataset(spec_dics, gen_nbrs=True, nbrlist_cutoff=5.0):

    print("Making dataset with %d species" % (len(spec_dics)))

    props_list = []
    nbr_list = []

    for j, spec_dic in enumerate(spec_dics):
        # treat each species' data like a regular dataset
        # and use it to generate neighbor lists

        dataset = Dataset(spec_dic.copy(), units='kcal/mol')
        mol_size = len(dataset.props["nxyz"][0])

        if gen_nbrs:
            dataset.generate_neighbor_list(cutoff=nbrlist_cutoff)

            # now combine the neighbor lists so that this set
            # of nxyz's can be treated like one big molecule

            nbrs = dataset.props['nbr_list']
            # number of atoms in the molecule
            new_nbrs = []

            # shift by i * mol_size for each conformer
            for i in range(len(nbrs)):
                new_nbrs.append(nbrs[i] + i * mol_size)

            # add to list of conglomerated neighbor lists
            nbr_list.append(torch.cat(new_nbrs))
            dataset.props.pop('nbr_list')

        # concatenate the nxyz's
        nxyz = np.concatenate([np.array(item) for item in spec_dic["nxyz"]]
                              ).reshape(-1, 4).tolist()

        # import pdb
        # pdb.set_trace()

        # add properties as necessary
        new_dic = {"mol_size": mol_size,
                   "nxyz": [nxyz],
                   "weights": torch.Tensor(spec_dic["weights"]
                                           ).reshape(-1, 1) / sum(
                       spec_dic["weights"]),
                   "degeneracy": torch.Tensor(spec_dic["degeneracy"]
                                              ).reshape(-1, 1),
                   "energy": torch.Tensor(spec_dic["energy"]
                                          ).reshape(-1, 1),
                   "num_atoms": [len(nxyz)]}

        new_dic.update({key: val[:1] for key, val in dataset.props.items(
        ) if key not in new_dic.keys()})
        props_list.append(new_dic)

        print("{} of {} complete".format(j, len(spec_dics)))

    print("Finalizing...")
    props_dic = concatenate_dict(*props_list)
    # make a combined dataset where the species look like they're
    # one big molecule
    big_dataset = Dataset(props_dic.copy(), units='kcal/mol')

    # give it the proper neighbor list
    if gen_nbrs:
        big_dataset.props['nbr_list'] = nbr_list

    print("Complete!")

    return big_dataset


def main(msg_file, dataset_path, num_specs, num_confs, max_atoms, smiles_csv):
    overall_dic = load_data(msg_file, num_specs, max_atoms, smiles_csv)
    spec_dics = convert_data(overall_dic, num_confs)
    dataset = make_nff_dataset(spec_dics=spec_dics,
                               gen_nbrs=True,
                               nbrlist_cutoff=5.0)
    dataset.save(dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--msg_file', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--num_specs', type=int)
    parser.add_argument('--smiles_csv', type=str)
    parser.add_argument('--num_confs', type=int)
    parser.add_argument('--max_atoms', type=int, default=MAX_ATOMS)
    arguments = parser.parse_args()

    try:
        main(**arguments.__dict__)
    except Exception as e:
        print(e)
        pdb.post_mortem()
