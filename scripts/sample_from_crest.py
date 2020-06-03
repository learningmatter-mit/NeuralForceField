import msgpack
import torch
import numpy as np
import argparse

from nff.data import Dataset, concatenate_dict

KEY_MAP = {"xyz": "nxyz", "boltzmannweight": "weights"}


def load_data(file, num_specs):

    file = "drugs_ensembles.msgpack"
    unpacker = msgpack.Unpacker(open(file, "rb"))
    spec_count = 0
    overall_dic = {}

    for chunk in unpacker:
        spec_count += len(chunk)
        overall_dic.update(chunk)
        if spec_count >= num_specs:
            break
    return overall_dic


def map_key(key):
    if key in KEY_MAP:
        return KEY_MAP[key]
    else:
        return key


def convert_data(overall_dic, num_confs):

    spec_dics = []
    for key, sub_dic in overall_dic.items():
        spec_dic = {map_key(key): val for key, val in sub_dic.items()
                    if key != "conformers"}
        spec_dic.update({map_key(key): [] for key
                         in sub_dic["conformers"].keys()})

        for conf in sub_dic["conformers"][:num_confs]:
            for key in conf.keys():
                spec_dic[map_key(key)].append(conf[key])
        spec_dics.append(spec_dic)

    return spec_dics


def make_nff_dataset(spec_dics, gen_nbrs=True, nbrlist_cutoff=5.0):

    print("Making dataset with %d species" % (len(spec_dics)))

    props_list = []
    nbr_list = []

    for spec_dic in spec_dics:
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
                   "nxyz": nxyz,
                   "weights": torch.Tensor(spec_dic["weights"]
                                           ).reshape(-1, 1),
                   "geom_ids": torch.Tensor(spec_dic["geom_ids"]
                                            ).reshape(-1, 1),
                   "degeneracy": torch.Tensor(spec_dic["degeneracy"]
                                              ).reshape(-1, 1),
                   "energy": torch.Tensor(spec_dic["energy"]
                                          ).reshape(-1, 1),
                   "num_atoms": [len(nxyz)]}

        new_dic.update({key: val[:1] for key, val in dataset.props.items(
        ) if key not in new_dic.keys()})
        props_list.append(new_dic)

    props_dic = concatenate_dict(*props_list)
    # make a combined dataset where the species look like they're
    # one big molecule
    big_dataset = Dataset(props_dic.copy(), units='kcal/mol')

    # give it the proper neighbor list
    if gen_nbrs:
        big_dataset.props['nbr_list'] = nbr_list

    return big_dataset


def main(msg_file, dataset_path, num_specs, num_confs):
    overall_dic = load_data(msg_file, num_specs)
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
    parser.add_argument('--num_confs', type=int)
    arguments = parser.parse_args()

    main(**arguments.__dict__)

