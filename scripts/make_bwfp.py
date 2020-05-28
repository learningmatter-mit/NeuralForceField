import os
import sys

sys.path.insert(0, "/home/saxelrod/repo/htvs/master/htvs")
sys.path.insert(0, "/home/saxelrod/repo/htvs/master/htvs/djangochem")
sys.path.insert(0, "/home/saxelrod/repo/nff/covid/NeuralForceField")

import django
os.environ["DJANGO_SETTINGS_MODULE"]="djangochem.settings.orgel"

from django.db import connections
import json
import argparse
import copy
import torch
import numpy as np

from nff.data.parallel import (split_dataset, rd_parallel,
                               summarize_rd, rejoin_props)
from nff.data import Dataset
from nff.data.features import (featurize_bonds, featurize_atoms)
from nff.data.features import add_model_fps


METHOD_NAME = 'gfn2-xtb'
METHOD_DESCRIP = 'Crest GFN2-xTB'
SPECIES_PATH = "/pool001/saxelrod/data_from_fock/data/covid_data/spec_ids.json"
GROUP_NAME = 'covid'
MODEL_PATH = "/pool001/saxelrod/data_from_fock/energy_model/best_model"
BASE_SAVE_PATH = "/pool001/saxelrod/data_from_fock/crest_fingerprint_datasets"

NUM_THREADS = 100
NUM_CONFS = 10
FP_LENGTH = 1024
SAVE_FEATURES = ['mean_e3fp', 'morgan', 'model_fp']
CSV_PROPS = ['sars_cov_one_pl_protease_active', 'ecoli_inhibitor', 'pseudomonas_active',
             'sars_cov_one_cl_protease_active', 'ensembleentropy', 'ensemblefreeenergy',
             'poplowestpct', 'totalconfs', 'ensembleenergy',
             'uniqueconfs', 'lowestenergy']


def get_rd_dataset(dataset,
                   thread_number,
                   num_procs=10,
                   base_save_path=BASE_SAVE_PATH,
                   model_path=None):

    if 'rd_mols' not in dataset.props:

        print("Featurizing dataset with {} parallel processes.".format(
            num_procs))
        datasets = split_dataset(dataset=dataset, num=num_procs)

        print("Converting xyz to RDKit mols...")
        datasets = rd_parallel(datasets)
        summarize_rd(new_sets=datasets, first_set=dataset)

        new_props = rejoin_props(datasets)
        dataset.props = new_props

    if 'atom_features' not in dataset.props:

        print("Featurizing bonds...")
        dataset = featurize_bonds(dataset)
        print("Completed featurizing bonds.")

        print("Featurizing atoms...")
        dataset = featurize_atoms(dataset)
        print("Completed featurizing atoms.")

        dataset.props["bonded_nbr_list"] = copy.deepcopy(
            dataset.props["bond_list"])
        dataset.props.pop("bond_list")

    if 'e3fp' not in dataset.props:

        print("Adding E3FP fingerprint...")
        dataset.add_e3fp(FP_LENGTH)
        print("Completed adding E3FP fingerprint.")

    if 'morgan' not in dataset.props:

        print("Adding Morgan fingerprint...")
        dataset.add_morgan(FP_LENGTH)
        print("Completed adding Morgan fingerprint.")

    if 'model_fp' not in dataset.props:

        print("Getting fingerprints from trained model...")
        add_model_fps(dataset, model_path)
        print("Finished getting fingerprints.")

    save_path = os.path.join(
        base_save_path, "crest_dset_{}.pth.tar".format(thread_number))
    dataset.save(save_path)

    return dataset


def get_bind_dataset(spec_ids,
                     batch_size=3,
                     geoms_per_spec=GEOMS_PER_SPEC,
                     method_name=METHOD_NAME,
                     method_descrip=METHOD_DESCRIP,
                     group_name=GROUP_NAME):

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
                                          bind_tags=[])

    print("Loader created.")
    connections.close_all()

    return dataset, loader


def dataset_getter(data_path,
                   num_threads,
                   thread_number,
                   all_spec_ids,
                   model_path):
    if os.path.isfile(data_path):
        rd_dataset = Dataset.from_file(data_path)
        rd_dataset = get_rd_dataset(rd_dataset,
                                    num_procs=10,
                                    thread_number=thread_number,
                                    model_path=model_path)

    else:
        spec_ids = get_subspec_ids(all_spec_ids=all_spec_ids,
                                   num_threads=num_threads,
                                   thread_number=thread_number)

        print("Got species IDs.")

        dataset, _ = get_bind_dataset(spec_ids)
        rd_dataset = get_rd_dataset(dataset,
                                    num_procs=10,
                                    thread_number=thread_number)
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
        text += "{},{}\n".format(smiles, prop)
    with open(csv_path, "w") as f:
        f.write(text)


def save_properties(rd_dataset, thread_number, base_path=BASE_SAVE_PATH):

    for prop_name in CSV_PROPS:

        if prop_name not in rd_dataset.props:
            continue

        # loop over idx because 'nan' is a float and so LongTensors
        # can't be concatenated with nan's

        good_idx = []
        props = []
        smiles_list = []

        for i, prop in enumerate(rd_dataset.props[prop_name]):
            if not torch.isnan(prop).item():
                good_idx.append(i)
                props.append(prop)
                smiles_list.append(rd_dataset.props["smiles"][i])

        # save

        csv_name = "{}_{}.csv".format(prop_name, str(thread_number))
        idx_name = "{}_{}_idx.json".format(prop_name, str(thread_number))

        csv_path = os.path.join(base_path, csv_name)
        idx_path = os.path.join(base_path, idx_name)

        with open(idx_path, "w") as f:
            json.dump(good_idx, f)

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

        save_pth = "{}_{}.npz".format(feature_name, str(thread_number))
        np.savez_compressed(save_pth, features=np_features)


def main(thread_number,
         num_confs=NUM_CONFS,
         num_threads=NUM_THREADS,
         model_path=MODEL_PATH,
         base_path=BASE_SAVE_PATH,
         species_path=SPECIES_PATH,
         prefix='combined'):

    print("Loading species ids...")

    with open(species_path, "r") as f:
        all_spec_ids = json.load(f)

    data_path = os.path.join(
        base_path, "{}_dset_{}.pth.tar".format(prefix, thread_number))

    rd_dataset = dataset_getter(data_path, num_threads, thread_number,
                                all_spec_ids, model_path)

    print("Saving save_properties...")
    save_properties(rd_dataset, thread_number, base_path)
    print("Finished saving properties.")

    print("Saving fingerprints...")
    save_features(rd_dataset, thread_number, base_path)
    print("Complete!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('thread_number', type=int, help='Thread number')
    parser.add_argument('num_threads', type=int, help='Number of threads')
    parser.add_argument('--num_confs', type=int, help='Number of conformers',
                        default=NUM_CONFS)
    parser.add_argument('--prefix', type=str, help='Fingerprint type',
                        default='combined')
    arguments = parser.parse_args()

    main(thread_number=arguments.thread_number,
         num_threads=arguments.num_threads,
         prefix=arguments.prefix)


