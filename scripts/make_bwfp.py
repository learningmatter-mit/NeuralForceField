from nff.uitls.data import from_db_pickle, get_bond_list
from nff.utils import data_to_yml
from nff.data.features import add_model_fps
from nff.data.features import (featurize_bonds, featurize_atoms)
from nff.data import Dataset
from nff.data.parallel import (split_dataset, rd_parallel,
                               import numpy as np
                               import torch
                               import copy
                               import argparse
                               import json
                               from django.db import connections
                               import django
                               import os
                               import sys

                               sys.path.insert(
                                   0, "/home/saxelrod/repo/htvs/master/htvs")
                               sys.path.insert(
                                   0, "/home/saxelrod/repo/htvs/master/htvs/djangochem")
                               sys.path.insert(
                                   0, "/home/saxelrod/repo/nff/covid/NeuralForceField")

                               os.environ["DJANGO_SETTINGS_MODULE"]="djangochem.settings.orgel"


                               summarize_rd, rejoin_props)

METHOD_NAME = 'gfn2-xtb'
METHOD_DESCRIP = 'Crest GFN2-xTB'
SPECIES_PATH = "/pool001/saxelrod/data_from_fock/data/covid_data/spec_ids.json"
GROUP_NAME = 'covid'
MODEL_PATH = "/pool001/saxelrod/data_from_fock/energy_model/best_model"
BASE_SAVE_PATH = ("/pool001/saxelrod/data_from_fock"
                  "/final_fingerprint_datasets")


NUM_THREADS = 100
NUM_CONFS = 10
FP_LENGTH = 1024
NUM_PROCS = 5
SAVE_FEATURES = ['mean_e3fp', 'morgan', 'model_fp']
CSV_PROPS = ['sars_cov_one_pl_protease_active', 'ecoli_inhibitor',
             'pseudomonas_active', 'sars_cov_one_cl_protease_active',
             'ensembleentropy', 'ensemblefreeenergy',
             'poplowestpct', 'totalconfs', 'ensembleenergy',
             'uniqueconfs', 'lowestenergy']


def get_rd_dataset(dataset,
                   thread_number,
                   num_procs=NUM_PROCS,
                   base_save_path=BASE_SAVE_PATH,
                   model_path=None,
                   get_model_fp=False,
                   no_features=False):

    if 'rd_mols' not in dataset.props and not no_features:

        print("Featurizing dataset with {} parallel processes.".format(
            num_procs))
        datasets = split_dataset(dataset=dataset, num=num_procs)

        print("Converting xyz to RDKit mols...")
        datasets = rd_parallel(datasets)
        summarize_rd(new_sets=datasets, first_set=dataset)

        new_props = rejoin_props(datasets)
        dataset.props = new_props

    if 'atom_features' not in dataset.props and not no_features:

        print("Featurizing bonds...")
        dataset = featurize_bonds(dataset)
        print("Completed featurizing bonds.")

        print("Featurizing atoms...")
        dataset = featurize_atoms(dataset)
        print("Completed featurizing atoms.")

        dataset.props["bonded_nbr_list"] = copy.deepcopy(
            dataset.props["bond_list"])
        dataset.props.pop("bond_list")

    if 'e3fp' not in dataset.props and not no_features:

        print("Adding E3FP fingerprint...")
        dataset.add_e3fp(FP_LENGTH)
        print("Completed adding E3FP fingerprint.")

    if 'morgan' not in dataset.props and not no_features:

        print("Adding Morgan fingerprint...")
        dataset.add_morgan(FP_LENGTH)
        print("Completed adding Morgan fingerprint.")

    if 'model_fp' not in dataset.props and get_model_fp and not no_features:

        print("Getting fingerprints from trained model...")
        add_model_fps(dataset, model_path)
        print("Finished getting fingerprints.")

    return dataset


def make_bond_nbrs(dataset, num_procs=5):

    print("Converting dataset xyz's to mols with {} parallel processes.".format(
        num_procs))
    datasets = split_dataset(dataset=dataset, num=num_procs)

    print("Converting xyz to RDKit mols...")
    datasets = rd_parallel(datasets)
    summarize_rd(new_sets=datasets, first_set=dataset)

    new_props = rejoin_props(datasets)
    dataset.props = new_props

    bond_dic = {}

    for rd_mols, smiles in zip(dataset.props["rd_mols"],
                           dataset.props["smiles"]):
        bond_dic[smiles] = []
        for mol in rd_mols:
            bond_list = get_bond_list(mol)
            bond_dic[smiles].append(bond_list)
        


def get_bind_dataset(spec_ids,
                     batch_size=3,
                     geoms_per_spec=NUM_CONFS,
                     method_name=METHOD_NAME,
                     method_descrip=METHOD_DESCRIP,
                     group_name=GROUP_NAME,
                     no_nbrs=False):

    django.setup()
    from neuralnet.utils.nff import create_bind_dataset

    print("Creating loader...")

    nbrlist_cutoff = 5.0
    num_workers = 2

    dataset, loader = create_bind_dataset(group_name=group_name,
                                          method_name=method_name,
                                          method_descrip=method_descrip,
                                          geoms_per_spec=geoms_per_spec,
                                          nbrlist_cutoff=nbrlist_cutoff,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          molsets=None,
                                          exclude_molsets=None,
                                          spec_ids=spec_ids,
                                          bind_tags=[],
                                          gen_nbrs=(not no_nbrs))

    print("Loader created.")
    connections.close_all()

    return dataset, loader


def dataset_getter(data_path,
                   num_threads,
                   thread_number,
                   all_spec_ids,
                   model_path,
                   get_model_fp,
                   no_features,
                   no_nbrs,
                   pickle_path=None,
                   nbrlist_cutoff=5.0,
                   num_procs=NUM_PROCS,
                   num_confs=NUM_CONFS):

    if os.path.isfile(data_path):
        dataset = Dataset.from_file(data_path)

    elif os.path.isfile(str(pickle_path)):
        dataset = from_db_pickle(pickle_path, nbrlist_cutoff)

    else:
        spec_ids = get_subspec_ids(all_spec_ids=all_spec_ids,
                                   num_threads=num_threads,
                                   thread_number=thread_number)
        dataset, _ = get_bind_dataset(spec_ids, no_nbrs=no_nbrs,
                                      geoms_per_spec=num_confs)

    rd_dataset = get_rd_dataset(dataset=dataset,
                                thread_number=thread_number,
                                num_procs=num_procs,
                                model_path=model_path,
                                get_model_fp=get_model_fp,
                                no_features=no_features)

    rd_dataset.save(data_path)

    return rd_dataset


def get_subspec_ids(all_spec_ids, num_threads, thread_number):

    chunk_size = int(len(all_spec_ids) / num_threads)
    start_idx = thread_number * chunk_size
    end_idx = (thread_number + 1) * chunk_size

    if thread_number == num_threads - 1:
        spec_ids = all_spec_ids[start_idx:]
    else:
        spec_ids = all_spec_ids[start_idx: end_idx]

    return spec_ids


def write_csv(smiles_list, props, prop_name, csv_path):

    text = "smiles,{}\n".format(prop_name)
    for smiles, prop in zip(smiles_list, props):
        if isinstance(prop, torch.Tensor):
            str_prop = str(prop.item())
        else:
            str_prop = str(prop)
        text += "{},{}\n".format(smiles, str_prop)
    with open(csv_path, "w") as f:
        f.write(text)


def save_properties(rd_dataset, thread_number, base_path=BASE_SAVE_PATH):

    for prop_name in CSV_PROPS:

        if prop_name not in rd_dataset.props:
            continue

        # loop over idx because 'nan' is a float and so LongTensors
        # can't be concatenated with nan's

        props = []
        smiles_list = []

        for i, prop in enumerate(rd_dataset.props[prop_name]):
            if not torch.isnan(prop).item():
                props.append(prop)
                smiles_list.append(rd_dataset.props["smiles"][i])

        # save

        csv_name = "{}_{}.csv".format(prop_name, str(thread_number))
        csv_path = os.path.join(base_path, csv_name)

        write_csv(smiles_list=smiles_list,
                  props=props,
                  prop_name=prop_name,
                  csv_path=csv_path)


def save_features(rd_dataset, thread_number, base_path=BASE_SAVE_PATH):

    for feature_name in SAVE_FEATURES:
        if feature_name not in rd_dataset.props:
            continue

        features = rd_dataset.props[feature_name]
        if isinstance(features, list):
            features = torch.stack(features)
        np_features = features.numpy()

        save_name = "{}_{}.npz".format(feature_name, str(thread_number))
        save_pth = os.path.join(base_path, save_name)
        np.savez_compressed(save_pth, features=np_features,
                            smiles=rd_dataset.props["smiles"])


def main(thread_number,
         num_confs=NUM_CONFS,
         num_threads=NUM_THREADS,
         model_path=MODEL_PATH,
         base_path=BASE_SAVE_PATH,
         species_path=SPECIES_PATH,
         prefix='combined',
         get_model_fp=False,
         no_features=False,
         no_nbrs=False,
         pickle_path=None):

    print("Loading species ids...")

    with open(species_path, "r") as f:
        all_spec_ids = json.load(f)

    data_path = os.path.join(
        base_path, "{}_dset_{}.pth.tar".format(prefix, thread_number))

    rd_dataset = dataset_getter(data_path=data_path,
                                num_threads=num_threads,
                                thread_number=thread_number,
                                all_spec_ids=all_spec_ids,
                                model_path=model_path,
                                get_model_fp=get_model_fp,
                                no_features=no_features,
                                no_nbrs=no_nbrs,
                                num_confs=num_confs,
                                pickle_path=pickle_path)

    if not no_features:

        print("Saving save_properties...")
        save_properties(rd_dataset, thread_number, base_path)
        print("Finished saving properties.")

        print("Saving fingerprints...")
        save_features(rd_dataset, thread_number, base_path)

    print("Saving as yml...")
    yml_path = os.path.join(
        base_path, "{}_dset_{}.yml".format(prefix, thread_number))
    data_to_yml.save_data(dataset=rd_dataset,
                          use_features=(not no_features),
                          save_path=yml_path)

    print("Complete!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--thread_number', type=int, help='Thread number')
    parser.add_argument('num_threads', type=int, help='Number of threads')
    parser.add_argument('--num_confs', type=int, help='Number of conformers',
                        default=NUM_CONFS)
    parser.add_argument('--prefix', type=str, help='Fingerprint type',
                        default='combined')
    parser.add_argument('--get_model_fp', action='store_true', default=False)
    parser.add_argument('--no_features', action='store_true', default=False)
    parser.add_argument('--no_nbrs', action='store_true', default=False)
    parser.add_argument('--base_path', type=str, default=BASE_SAVE_PATH)
    parser.add_argument('--picke_path', type=str, default=None)

    arguments = parser.parse_args()

    main(thread_number=arguments.thread_number,
         num_threads=arguments.num_threads,
         prefix=arguments.prefix,
         get_model_fp=arguments.get_model_fp,
         no_features=arguments.no_features,
         no_nbrs=arguments.no_nbrs,
         base_path=arguments.base_path,
         num_confs=arguments.num_confs,
         pickle_path=arguments.pickle_path)
