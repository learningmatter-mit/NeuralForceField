import sys

sys.path.insert(0, "/home/saxelrod/repo/htvs/master/htvs")
sys.path.insert(0, "/home/saxelrod/repo/htvs/master/htvs/djangochem")
sys.path.insert(0, "/home/saxelrod/repo/nff/covid/NeuralForceField")


import argparse

from nff.data.parallel import (split_dataset, rd_parallel,
                               summarize_rd, rejoin_props)
from nff.utils.data import (from_db_pickle, get_bond_list, split_confs)
import pickle
import time

def main(pickle_path, save_path, num_procs=5, nbrlist_cutoff=5):

    start = time.time()

    print("Loading dataset and generating non-bonded neighbour list...")
    dataset = from_db_pickle(pickle_path, nbrlist_cutoff)

    print("Converting dataset xyz's to mols with {} parallel processes.".format(
        num_procs))
    datasets = split_dataset(dataset=dataset, num=num_procs)

    print("Converting xyz to RDKit mols...")
    datasets = rd_parallel(datasets)
    summarize_rd(new_sets=datasets, first_set=dataset)

    new_props = rejoin_props(datasets)
    dataset.props = new_props

    combined_dic = {}

    for dic in dataset:

        smiles = dic["smiles"]
        rd_mols = dic["rd_mols"]
        combined_dic[smiles] = {"bonds": []}

        for mol in rd_mols:
            bond_list = get_bond_list(mol)
            combined_dic[smiles]["bonds"].append(bond_list)
        
        split_nbrs = split_confs(dic)
        nbr_name = "nbrs_%.dA" % nbrlist_cutoff
        combined_dic[smiles][nbr_name] = split_nbrs

    print("Saving...")

    with open(save_path, "wb") as f:
        pickle.dump(combined_dic, f)

    end = time.time()
    hours = (end - start) / 3600
    print("Finished in %.2f hours" % hours)
    print("Complete!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_path', type=str, help='Pickle path')
    parser.add_argument('--save_path', type=str, help='Save path for nbr list')
    parser.add_argument('--num_procs', type=int, help='Parallel processes for xyz2mol',
                        default=5)
    parser.add_argument('--nbrlist_cutoff', type=int, help='Neighbour list cutoff',
                        default=5)


    arguments = parser.parse_args()

    main(pickle_path=arguments.pickle_path, 
        save_path=arguments.save_path, 
        num_procs=arguments.num_procs,
         nbrlist_cutoff=arguments.nbrlist_cutoff)


