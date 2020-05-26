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
import numpy as np
from e3fp.pipeline import fprints_from_mol


from nff.utils.cuda import batch_to
from nff.train.builders.model import load_model
from nff.data.parallel import (split_dataset, rd_parallel,
                               summarize_rd, rejoin_props)
from nff.data import Dataset

METHOD_NAME = 'gfn2-xtb'
METHOD_DESCRIP = 'Crest GFN2-xTB'
SPECIES_PATH = "/pool001/saxelrod/data_from_fock/data/covid_data/spec_ids.json"
GEOMS_PER_SPEC = 10
GROUP_NAME = 'covid'
MODEL_PATH = "/pool001/saxelrod/data_from_fock/energy_model/best_model"
BASE_SAVE_PATH = "/pool001/saxelrod/data_from_fock/crest_fingerprint_datasets"
NUM_THREADS = 100
COVID_TAG = "sars_cov_one_cl_protease_active"


def get_rd_dataset(dataset,
                   thread_number,
                   num_procs=10,
                   base_save_path=BASE_SAVE_PATH):

    print("Featurizing dataset with {} parallel processes.".format(
        num_procs))
    datasets = split_dataset(dataset=dataset, num=num_procs)

    print("Converting xyz to RDKit mols...")
    datasets = rd_parallel(datasets)
    summarize_rd(new_sets=datasets, first_set=dataset)

    new_props = rejoin_props(datasets)
    dataset.props = new_props

    save_path = os.path.join(
        base_save_path, "crest_dset_{}.pth.tar".format(thread_number))
    dataset.save(save_path)

    return dataset


def get_e3fp(rd_dataset, num_confs):

    bwfp_dic = {}

    for batch in rd_dataset:

        mols = batch["rd_mols"]
        weights = batch["weights"]
        smiles = batch["smiles"]
        sorted_idx = np.argsort(-weights.numpy()).reshape(-1)[:num_confs]
        fps = []

        for idx in sorted_idx:
            
            weight = weights[idx] / weights[sorted_idx].sum()
            mol = mols[idx]

            mol.SetProp("_Name", smiles)
            fprint_params = {"bits": 2048}
            fp = fprints_from_mol(mol, fprint_params=fprint_params)
            fp_array = np.zeros(len(fp[0]))
            indices = fp[0].indices
            fp_array[indices] = 1

            fps.append(fp_array * weight.item())
        bwfp_dic[smiles] = np.array(fps).mean(0).tolist()

    return bwfp_dic


def get_loader(spec_ids,
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
                                          bind_tags=[COVID_TAG])

    print("Loader created.")
    connections.close_all()

    return dataset, loader


def main_e3fp(thread_number,
              num_confs,
              num_threads=NUM_THREADS,
              base_path=BASE_SAVE_PATH,
              species_path=SPECIES_PATH):

    print("Loading species ids...")

    with open(species_path, "r") as f:
        all_spec_ids = json.load(f)

    data_path = os.path.join(
        base_path, "crest_dset_{}.pth.tar".format(thread_number))

    if os.path.isfile(data_path):
       	rd_dataset = Dataset.from_file(data_path)
        # props = {key: val[:10] for key, val in rd_dataset.props.items()}
        # rd_dataset.props = props
    else:
        spec_ids = get_subspec_ids(all_spec_ids=all_spec_ids,
                                   num_threads=num_threads,
                                   thread_number=thread_number)

        print("Got species IDs.")

        dataset, _ = get_loader(spec_ids)
        rd_dataset = get_rd_dataset(dataset,
                                    num_procs=10,
                                    thread_number=thread_number)

    e3fp_dic = get_e3fp(rd_dataset, num_confs)
    save_path = os.path.join(
        base_path, "e3fp_bwfp_{}_{}_confs.json".format(thread_number,
       num_confs))

    print("Saving fingerprints...")

    with open(save_path, "w") as f:
        json.dump(e3fp_dic, f, indent=4, sort_keys=True)

    print("Complete!")


def get_batch_fps(model_path, loader, device=0):

    print("Getting fingerprints...")

    model = load_model(model_path)
    dic = {}
    loader_len = len(loader)

    for i, batch in enumerate(loader):

        batch = batch_to(device)

        conf_fps, _ = model.embedding_forward(batch
                                           ).detach().cpu().numpy().tolist()
        smiles_list = batch['smiles']

        assert len(smiles_list) == len(conf_fps)

        dic.update({smiles: conf_fp
                    for smiles, conf_fp in zip(smiles_list, conf_fps)})

        pct = int(i / loader_len) * 100
        print("%d%% done" % pct)

    print("Finished getting fingerprints.")

    return dic


def get_subspec_ids(all_spec_ids, num_threads, thread_number):

    chunk_size = int(len(all_spec_ids) / num_threads)
    start_idx = thread_number * chunk_size
    end_idx = (thread_number + 1) * chunk_size

    if thread_number == num_threads - 1:
        spec_ids = all_spec_ids[start_idx:]
    else:
        spec_ids = all_spec_ids[start_idx: end_idx]

    return spec_ids


def main(thread_number,
         num_threads=NUM_THREADS,
         model_path=MODEL_PATH,
         base_path=BASE_SAVE_PATH,
         species_path=SPECIES_PATH):

    print("Loading species ids...")

    with open(species_path, "r") as f:
        all_spec_ids = json.load(f)

    spec_ids = get_subspec_ids(all_spec_ids=all_spec_ids,
                               num_threads=num_threads,
                               thread_number=thread_number)

    print("Got species IDs.")

    loader = get_loader(spec_ids)
    fp_dic = get_batch_fps(model_path, loader)

    save_path = os.path.join(
        base_path, "bwfp_{}_schnet.json".format(thread_number))

    print("Saving fingerprints...")

    with open(save_path, "w") as f:
        json.dumps(fp_dic, f, indent=4, sort_keys=True)

    print("Complete!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('thread_number', type=int, help='Thread number')
    parser.add_argument('num_threads', type=int, help='Number of threads')
    parser.add_argument('--fp_type', type=str, help='Fingerprint type',
                        default='e3fp')
    parser.add_argument('--num_confs', type=int, help='Number of conformers',
                        default=10)
    arguments = parser.parse_args()

    if arguments.fp_type == 'e3fp':
        main_e3fp(thread_number=arguments.thread_number,
                  num_threads=arguments.num_threads,
                  num_confs=arguments.num_confs)
    else:
        main(thread_number=arguments.thread_number,
             num_threads=arguments.num_threads)
