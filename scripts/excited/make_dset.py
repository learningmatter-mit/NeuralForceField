import sys
import os
import django

HOME = os.environ["HOME"]
HTVS_DIR = os.environ.get("HTVSDIR", f"{HOME}/htvs")
DJANGOCHEMDIR = os.environ.get("DJANGOCHEMDIR", f"{HOME}/htvs/djangochem")
NFFDIR = os.environ.get("NFFDIR",
                        f"{HOME}/Repo/projects/master/NeuralForceField")

sys.path.insert(0, HTVS_DIR)
sys.path.insert(0, DJANGOCHEMDIR)
sys.path.insert(0, NFFDIR)


os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.orgel"
django.setup()

import argparse

from nff.data import Dataset, concatenate_dict, split_train_validation_test
from pgmols.models import Calc
from jobs.models import Job, JobConfig
from tqdm import tqdm
import torch
import json


CONFIG_DIC = {"bhhlyp_6-31gs_sf_engrad":
              {"name": "sf_tddft_bhhlyp",
               "description": "GAMESS bhhlyp/6-31G* spin flip tddft"},
              "bhhlyp_6-31gs_sf_hop":
              {"name": "sf_tddft_bhhlyp",
               "description": "GAMESS bhhlyp/6-31G* spin flip tddft"}}

CHUNK_SIZE = 1000


def trim_overall(overall_dict, required_keys):

    keys = list(overall_dict.keys())
    for key in keys:
        sub_dic = overall_dict[key]
        keep = True
        for req in required_keys:
            if req not in sub_dic:
                keep = False
                break
        if not keep:
            overall_dict.pop(key)
    return overall_dict


def save_dset(overall_dict,
              required_keys,
              save_dir,
              idx,
              val_size,
              test_size,
              seed):

    overall_dict = trim_overall(overall_dict, required_keys)
    props = concatenate_dict(*list(overall_dict.values()))
    dset = Dataset(props=props, units='atomic')
    save_folder = os.path.join(save_dir, str(idx))
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    splits = split_train_validation_test(dset,
                                         val_size=val_size,
                                         test_size=test_size,
                                         seed=seed)
    names = ["train", "val", "test"]
    for split, name in zip(splits, names):
        save_path = os.path.join(save_folder, f"{name}.pth.tar")
        split.save(save_path)


def parse_optional(max_geoms,
                   max_geoms_per_dset,
                   molsets):
    if max_geoms is None:
        max_geoms = float('inf')
    if molsets is None:
        molsets = []
    if max_geoms_per_dset is None:
        max_geoms_per_dset = max_geoms

    return max_geoms, max_geoms_per_dset, molsets


def get_job_pks(method_name,
                method_description):

    config_names = [key for key, val in CONFIG_DIC.items() if
                    val['name'] == method_name and
                    val['description'] == method_description]
    job_pks = list(JobConfig.objects.filter(name__in=config_names)
                   .order_by("?").values_list('job__pk', flat=True))

    return job_pks


def get_job_query(job_pks, chunk_size, group_name, molsets):

    todo = []
    for _ in range(chunk_size):
        if not job_pks:
            break
        todo += [job_pks.pop()]
    job_query = Job.objects.filter(id__in=todo,
                                   status='done',
                                   group__name=group_name)
    for molset in molsets:
        job_query = job_query.filter(
            parentgeom__species__mol__sets__name=molset)
    return job_query


def update_overall(overall_dict, calc_pk):
    calc = Calc.objects.get(pk=calc_pk)
    geom = calc.geoms.first()
    geom_id = geom.id
    props = calc.props

    if geom_id not in overall_dict:
        overall_dict[geom_id] = {"nxyz": geom.xyz,
                                 "smiles": geom.species.smiles}

    if props is None:
        jacobian = calc.jacobian
        if not jacobian:
            return overall_dict
        forces = jacobian.forces
        overall_dict[geom_id].update({"energy_0_grad":
                                      (-torch.Tensor(forces)).tolist(),
                                      "geom_id": geom_id})
    else:
        force_1 = props['excitedstates'][0]['forces']
        overall_dict[geom_id].update(
            {"energy_0": props['totalenergy'],
             "energy_1": props['excitedstates'][0]['energy'],
             "energy_1_grad": (-torch.Tensor(force_1)).tolist()
             }
        )
    return overall_dict


def main(group_name,
         method_name,
         method_description,
         required_keys,
         max_geoms,
         max_geoms_per_dset,
         save_dir,
         val_size=0.1,
         test_size=0.1,
         split_seed=0,
         molsets=None,
         chunk_size=CHUNK_SIZE,
         **kwargs):

    max_geoms, max_geoms_per_dset, molsets = parse_optional(
        max_geoms=max_geoms,
        max_geoms_per_dset=max_geoms_per_dset,
        molsets=molsets)

    job_pks = get_job_pks(method_name=method_name,
                          method_description=method_description)

    geom_count = 0
    i = 0

    overall_dict = {}

    while job_pks:
        print("%d remaining..." % (len(job_pks)))

        job_query = get_job_query(job_pks=job_pks,
                                  chunk_size=chunk_size,
                                  group_name=group_name,
                                  molsets=molsets)

        calc_pks = list(job_query.values_list('childcalcs', flat=True))
        for calc_pk in tqdm(calc_pks):

            overall_dict = update_overall(overall_dict=overall_dict,
                                          calc_pk=calc_pk)

            if len(overall_dict) >= max_geoms_per_dset:
                save_dset(overall_dict=overall_dict,
                          required_keys=required_keys,
                          save_dir=save_dir,
                          idx=i,
                          val_size=val_size,
                          test_size=test_size,
                          seed=split_seed)

                geom_count += len(overall_dict)
                i += 1
                overall_dict = {}

            if geom_count >= max_geoms:
                break

    if not overall_dict:
        return

    save_dset(overall_dict=overall_dict,
              required_keys=required_keys,
              save_dir=save_dir,
              idx=i,
              val_size=val_size,
              test_size=test_size,
              seed=split_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        help=('The path with to the config file'),
                        default='job_info.json')
    args = parser.parse_args()
    config_path = args.config_file
    with open(config_path, 'r') as f:
        kwargs = json.load(f)

    main(**kwargs)
