import os
import django

os.environ["DJANGO_SETTINGS_MODULE"]="djangochem.settings.orgel"
django.setup()

import json
import argparse

from nff.utils.cuda import batch_to
from nff.train.builders.model import load_model
from neuralnet.utils.nff import create_bind_dataset

METHOD_NAME = 'gfn2-xtb'
METHOD_DESCRIP = 'Crest GFN2-xTB'
SPECIES_PATH = "/pool001/saxelrod/data_from_fock/data/covid_data/spec_ids.json"
GEOMS_PER_SPEC = 10
GROUP_NAME = 'covid'
METHOD_NAME = 'molecular_mechanics_mmff94'
METHOD_DESCRIP = 'MMFF conformer.'
MODEL_PATH = "/pool001/saxelrod/data_from_fock/energy_model/best_model"
BASE_SAVE_PATH = "/pool001/saxelrod/data_from_fock/fingerprint_datasets"
NUM_THREADS = 100

def get_loader(spec_ids,
               geoms_per_spec=GEOMS_PER_SPEC,
               method_name=METHOD_NAME,
               method_descrip=METHOD_DESCRIP,
               group_name=GROUP_NAME):
    
    print("Creating loader...")

    nbrlist_cutoff = 5.0
    batch_size = 3
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
                                          spec_ids=spec_ids)

    print("Loader created.")

    return loader


def get_batch_fps(model_path, loader, device=0):

    print("Getting fingerprints...")

    model = load_model(model_path)
    dic = {}
    loader_len = len(loader)

    for i, batch in enumerate(loader):

        batch = batch_to(device)

        conf_fps = model.embedding_forward(batch
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

    spec_ids = get_subspec_ids(all_spec_ids=all_spec_ids, num_threads=num_threads,
                               thread_number=thread_number)

    print("Got species IDs.")

    loader = get_loader(spec_ids)
    fp_dic = get_batch_fps(model_path, loader)

    save_path = os.path.join(base_path, "bwfp_{}.json".format(thread_number))

    print("Saving fingerprints...")

    with open(save_path, "w") as f:
        json.dumps(fp_dic, f, indent=4, sort_keys=True)

    print("Complete!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('thread_number', type=int, help='Thread number')
    arguments = parser.parse_args()

    main(thread_number=arguments.thread_number)

