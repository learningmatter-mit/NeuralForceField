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

import json
import torch
from tqdm import tqdm
from analysis.metalation_energy import custom_stoich
from jobs.models import Job, JobConfig
from pgmols.models import Calc
from nff.data import Dataset, concatenate_dict, split_train_validation_test
import argparse


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


def en_trim(good_idx_dic,
            sub_dic,
            key,
            max_std_en,
            max_val_en):
    val = torch.stack(sub_dic[key])
    std = val.std()
    mean = val.mean()

    bad_idx = ((abs(val - mean) > max_std_en * std).nonzero()
               .reshape(-1).tolist())

    for i in bad_idx:
        good_idx_dic[i] *= 0

    bad_idx = ((abs(val - mean) > max_val_en).nonzero()
               .reshape(-1).tolist())

    for i in bad_idx:
        good_idx_dic[i] *= 0
    return good_idx_dic


def grad_trim(good_idx_dic,
              sub_dic,
              key,
              max_std_force,
              max_val_force):

    val = torch.stack(sub_dic[key])
    std = val.std(0)
    mean = val.mean(0)

    bad_idx = ((abs(val - mean) > max_std_force * std).nonzero()
               [:, 0]).tolist()

    for i in bad_idx:
        good_idx_dic[i] *= 0

    bad_idx = ((abs(val - mean) > max_val_force).nonzero()
               [:, 0]).tolist()

    for i in bad_idx:
        good_idx_dic[i] *= 0
    return good_idx_dic


def init_outlier_dic(dset, en_keys, grad_keys):

    all_smiles = list(set([rm_stereo(i) for i in dset.props['smiles']]))
    en_outlier_dic = {smiles: {key: [] for key in ['idx', *en_keys]} for smiles
                      in all_smiles}
    grad_outlier_dic = {smiles: {key: [] for key in ['idx', *grad_keys]}
                        for smiles in all_smiles}
    for i, batch in enumerate(dset):

        smiles = rm_stereo(batch['smiles'])

        en_outlier_dic[smiles]['idx'].append(i)
        grad_outlier_dic[smiles]['idx'].append(i)

        for key in en_keys:
            en_outlier_dic[smiles][key].append(batch[key])
        for key in grad_keys:
            grad_outlier_dic[smiles][key].append(batch[key])

    return en_outlier_dic, grad_outlier_dic


def remove_outliers(dset,
                    max_std_en,
                    max_std_force,
                    max_val_en,
                    max_val_force):

    en_keys = [key for key in dset.props.keys() if 'energy' in key
               and 'grad' not in key]
    grad_keys = [key for key in dset.props.keys() if 'energy' in key
                 and 'grad' in key]

    en_outlier_dic, grad_outlier_dic = init_outlier_dic(
        dset=dset,
        en_keys=en_keys,
        grad_keys=grad_keys)

    good_idx_dic = {i: 1 for i in range(len(dset))}

    for smiles, sub_dic in en_outlier_dic.items():
        for key in en_keys:
            good_idx_dic = en_trim(good_idx_dic=good_idx_dic,
                                   sub_dic=sub_dic,
                                   key=key,
                                   max_std_en=max_std_en,
                                   max_val_en=max_val_en)

    for smiles, sub_dic in grad_outlier_dic.items():
        for key in grad_keys:
            good_idx_dic = grad_trim(good_idx_dic=good_idx_dic,
                                     sub_dic=sub_dic,
                                     key=key,
                                     max_std_force=max_std_force,
                                     max_val_force=max_val_force)

    final_idx = torch.LongTensor([i for i, val in good_idx_dic.items()
                                  if val == 1])

    for key, val in dset.props.items():
        if isinstance(val, list):
            dset.props[key] = [val[i] for i in final_idx]
        else:
            dset.props[key] = val[final_idx]

    return dset


def save_dset(overall_dict,
              required_keys,
              save_dir,
              idx,
              val_size,
              test_size,
              seed,
              max_std_en,
              max_std_force,
              max_val_en,
              max_val_force):

    overall_dict = trim_overall(overall_dict, required_keys)
    props = concatenate_dict(*list(overall_dict.values()))
    dset = Dataset(props=props, units='atomic')
    save_folder = os.path.join(save_dir, str(idx))
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    dset = remove_outliers(dset=dset,
                           max_std_en=max_std_en,
                           max_std_force=max_std_force,
                           max_val_en=max_val_en,
                           max_val_force=max_val_force)

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


def update_stoich(stoich_dict, geom, custom_name):
    formula = geom.stoichiometry.formula
    if formula in stoich_dict:
        return stoich_dict, formula

    stoich_en = custom_stoich(formula, custom_name)
    stoich_dict[formula] = stoich_en

    return stoich_dict, formula


def update_overall(overall_dict,
                   stoich_dict,
                   calc_pk,
                   custom_name):

    calc = Calc.objects.filter(pk=calc_pk).first()
    if not calc:
        return overall_dict, stoich_dict

    geom = calc.geoms.first()
    geom_id = geom.id
    props = calc.props

    if geom_id not in overall_dict:
        overall_dict[geom_id] = {"nxyz": geom.xyz,
                                 "smiles": geom.species.smiles}

    stoich_dict, formula = update_stoich(stoich_dict=stoich_dict,
                                         geom=geom,
                                         custom_name=custom_name)
    stoich_en = stoich_dict[formula]

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
            {"energy_0": props['totalenergy'] - stoich_en,
             "energy_1": props['excitedstates'][0]['energy'] - stoich_en,
             "energy_1_grad": (-torch.Tensor(force_1)).tolist()
             }
        )
    return overall_dict, stoich_dict


def rm_stereo(smiles):
    new_smiles = smiles.replace("/", "").replace("\\", "")
    return new_smiles


def main(group_name,
         method_name,
         method_description,
         required_keys,
         max_geoms,
         max_geoms_per_dset,
         save_dir,
         custom_name,
         max_std_en,
         max_std_force,
         max_val_en,
         max_val_force,
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
    stoich_dict = {}

    while job_pks:
        print("%d remaining..." % (len(job_pks)))

        job_query = get_job_query(job_pks=job_pks,
                                  chunk_size=chunk_size,
                                  group_name=group_name,
                                  molsets=molsets)

        calc_pks = list(job_query.values_list('childcalcs', flat=True))
        for calc_pk in tqdm(calc_pks):

            overall_dict, stoich_dict = update_overall(
                overall_dict=overall_dict,
                stoich_dict=stoich_dict,
                calc_pk=calc_pk,
                custom_name=custom_name)

            if len(overall_dict) >= max_geoms_per_dset:

                save_dset(overall_dict=overall_dict,
                          required_keys=required_keys,
                          save_dir=save_dir,
                          idx=i,
                          val_size=val_size,
                          test_size=test_size,
                          seed=split_seed,
                          max_std_en=max_std_en,
                          max_std_force=max_std_force,
                          max_val_en=max_val_en,
                          max_val_force=max_val_force
                          )

                geom_count += len(overall_dict)
                i += 1
                overall_dict = {}

            if geom_count >= max_geoms:
                break
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
