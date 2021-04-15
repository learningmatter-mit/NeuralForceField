import os
import django
import sys

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
import re
from rdkit import Chem
from sklearn import linear_model
import numpy as np
from django.contrib.auth.models import Group
from nff.data import Dataset, concatenate_dict, split_train_validation_test
from pgmols.models import Calc, Geom, GeomSet, Species, Method
from jobs.models import Job, JobConfig
from analysis.metalation_energy import (custom_stoich, stoich_energy,
                                        REFERENCE_PROJECT, DEFAULT_PROJECT)
from tqdm import tqdm
import torch
import json


CONFIG_DIC = {"bhhlyp_6-31gs_sf_tddft_engrad":
              {
                  "name": "sf_tddft_bhhlyp",
                  "description": "GAMESS bhhlyp/6-31G* spin flip tddft"
              },
              "bhhlyp_6-31gs_sf_hop":
              {
                  "name": "sf_tddft_bhhlyp",
                  "description": "GAMESS bhhlyp/6-31G* spin flip tddft"
              },
              "bhhlyp_6-31gs_sf_tddft_nacv_qchem":
              {
                  "name": "sf_tddft_bhhlyp",
                  "description": "QChem bhhlyp/6-31gs SF-TDDFT"
              },
              "bhhlyp_6-31gs_sf_tddft_engrad_qchem":
              {
                  "name": "sf_tddft_bhhlyp",
                  "description": "QChem bhhlyp/6-31gs SF-TDDFT"
              },
              "wb97xd_def2svp_engrad_orca": {
                  "name": "dft_hyb_wb97xd3",
                  "description": "ORCA wb97xd/def2-svp"
              },
              "wb97xd_def2svp_tda_tddft_engrad_orca": {
                  "name": "tddft_tda_hyb_wb97xd3",
                  "description": "ORCA wb97xd/def2-svp TDDFT (TDA)"
              }}

CHUNK_SIZE = 1000
EV_TO_AU = 0.0367493


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
            continue
        extra_props = [i for i in sub_dic.keys()
                       if i not in required_keys]
        for extra in extra_props:
            overall_dict[key].pop(extra)

    return overall_dict


def query_spec(spec,
               only_neutral,
               method_name,
               method_descrip,
               other_method_name,
               other_method_descrip):

    mol = Chem.MolFromSmiles(spec.smiles)
    mol = Chem.AddHs(mol)
    charge = Chem.rdmolops.GetFormalCharge(mol)
    if charge != 0 and only_neutral:
        return
    convg_geoms = spec.geom_set.filter(converged=True,
                                       parentjob__config__name__contains='opt')

    # we don't want a species' lowest energy geometry
    # if it isn't reasonably close to being converged,
    # which will be true only if there is some converged
    # geom somewhere for the species

    if not convg_geoms:
        return

    geoms = Geom.objects.filter(species=spec)
    calc_pks = geoms.values_list('calcs__pk')
    calcs = Calc.objects.filter(pk__in=calc_pks,
                                method__name=method_name,
                                method__description=method_descrip,
                                props__totalenergy__isnull=False)

    if all([other_method_name is not None,
            other_method_descrip is not None]):

        other_method = Method.objects.get(name=other_method_name,
                                          description=other_method_descrip)
        calcs = calcs.filter(geoms__calcs__method=other_method)

    # calc = (calcs.filter(geoms__isnull=False)
    #         .order_by("props__totalenergy").first())

    calc = (calcs.order_by("props__totalenergy").first())

    if not calc:
        return

    geom = calc.geoms.first()
    energy = calc.props['totalenergy']
    formula = spec.stoichiometry.formula

    return geom, energy, formula


def query_converged(group_name,
                    method_name,
                    method_descrip,
                    max_num,
                    other_method_name=None,
                    other_method_descrip=None,
                    only_neutral=True):
    """
    Get geoms that are converged or close to converged for a method
    """

    specs = Species.objects.filter(group__name=group_name)

    ####
    # specs = specs.filter(smiles='c1ccc2c(c1)CSc1ccccc1/N=N/2')
    ####

    energy_dic = {}
    geom_dic = {}
    i = 0

    for spec in tqdm(specs):
        out = query_spec(spec,
                         only_neutral,
                         method_name,
                         method_descrip,
                         other_method_name,
                         other_method_descrip)
        if out is None:
            continue

        geom, energy, formula = out

        if formula not in energy_dic:
            energy_dic[formula] = []
        if formula not in geom_dic:
            geom_dic[formula] = []

        energy_dic[formula].append(energy)
        geom_dic[formula].append(geom)

        i += 1
        if i >= max_num:
            break

    min_idx = {key: np.argmin(val) for key, val in energy_dic.items()}
    energy_dic = {key: val[min_idx[key]] for key, val in energy_dic.items()}
    geom_dic = {key: val[min_idx[key]] for key, val in geom_dic.items()}

    return energy_dic, geom_dic


def get_atom_count(formula):
    dictio = dict(re.findall('([A-Z][a-z]?)([0-9]*)', formula))
    for key, val in dictio.items():
        dictio[key] = int(val) if val.isdigit() else 1

    return dictio


def all_atoms(ground_en):
    atom_list = []
    for formula in ground_en.keys():
        dictio = get_atom_count(formula)
        atom_list += list(dictio.keys())
    atom_list = list(set(atom_list))

    return atom_list


def reg_atom_count(formula, atoms):
    dictio = get_atom_count(formula)
    count_array = np.array([dictio.get(atom, 0) for atom in atoms])

    return count_array


def get_stoich_data(group_name,
                    method_name,
                    method_descrip,
                    other_method_name=None,
                    other_method_descrip=None,
                    max_num=float('inf')):

    ground_en, geom_dic = query_converged(
        group_name=group_name,
        method_name=method_name,
        method_descrip=method_descrip,
        max_num=max_num,
        other_method_name=other_method_name,
        other_method_descrip=other_method_descrip)

    atoms_list = all_atoms(ground_en)
    formulas = list(ground_en.keys())
    x_in = np.stack([reg_atom_count(formula, atoms_list)
                     for formula in formulas])
    y_out = np.array([[ground_en[formula]] for formula in formulas])

    return x_in, y_out, atoms_list, ground_en, geom_dic


def fit_stoich(x_in,
               y_out,
               atoms_list):

    clf = linear_model.LinearRegression()
    clf.fit(x_in, y_out)
    pred = (clf.coef_ * x_in + clf.intercept_).sum(-1)
    err = abs(pred - y_out).mean() * 627.5
    print(("MAE between target energy and stoich "
           "energy is %.3e kcal/mol" % err))
    fit_dic = {atom: coef for atom, coef in zip(
        atoms_list, clf.coef_.reshape(-1))}
    stoich_dict = {**fit_dic, "offset": clf.intercept_.item()}

    return stoich_dict


def ens_from_geoms(geom_dic,
                   method_name,
                   method_descrip):

    energy_dic = {}
    for formula, geom in geom_dic.items():
        calc = geom.calcs.filter(method__name=method_name,
                                 method__description=method_descrip,
                                 props__totalenergy__isnull=False).first()
        energy = calc.props['totalenergy']
        energy_dic[formula] = energy

    return energy_dic


def get_compare_ens(geom_dic,
                    main_en_dic,
                    formulas,
                    compare_method_name,
                    compare_method_descrip,
                    compare_stoich_name):

    compare_energies = ens_from_geoms(geom_dic=geom_dic,
                                      method_name=compare_method_name,
                                      method_descrip=compare_method_descrip)
    compare_stoichs = {formula: custom_stoich(formula=formula,
                                              custom_name=compare_stoich_name)
                       for formula in formulas}

    # gamess crude energy
    compare_en = np.array([compare_energies[formula] for formula in formulas])
    # gamess ref energy
    compare_ref = np.array([compare_stoichs[formula] for formula in formulas])
    # q-chem energy
    main_en = np.array([main_en_dic[formula] for formula in formulas])

    # pred = qchem_en - qchem_ref
    # targ = gamess en - gamess ref
    # --> rearrange so qchem_ref = -gamess en + gamess ref + qchem_en

    target = -compare_en + compare_ref + main_en

    return target


def lin_reg_stoich(formula, atom_dict):
    dictio = dict(re.findall('([A-Z][a-z]?)([0-9]*)', formula))
    energy = atom_dict['offset']
    for key, val in dictio.items():
        if val.isdigit():
            count = int(val)
        else:
            count = 1

        energy += count * atom_dict[key]

    return energy


def match_stoich(geom_dic,
                 main_en_dic,
                 compare_method_name,
                 compare_method_descrip,
                 compare_stoich_name):
    """
    Choose the stoich energies so that the energies of the converged
    geoms, minus stoich, are as close as possible to their GAMESS 
    values.
    """

    formulas = list(geom_dic.keys())
    y_out = get_compare_ens(geom_dic=geom_dic,
                            main_en_dic=main_en_dic,
                            formulas=formulas,
                            compare_method_name=compare_method_name,
                            compare_method_descrip=compare_method_descrip,
                            compare_stoich_name=compare_stoich_name)

    atoms_list = all_atoms(main_en_dic)
    x_in = np.stack([reg_atom_count(formula, atoms_list)
                     for formula in formulas])

    stoich_dict = fit_stoich(x_in=x_in,
                             y_out=y_out,
                             atoms_list=atoms_list)

    return stoich_dict


def get_stoich_save(job_dir,
                    method_name,
                    method_descrip,
                    other_method_name,
                    other_method_descrip):

    main_pk = Method.objects.get(name=method_name,
                                 description=method_descrip).pk
    name = f"lin_stoich_{main_pk}"
    if all([other_method_name is not None,
            other_method_descrip is not None]):
        other_pk = Method.objects.get(name=other_method_name,
                                      description=other_method_descrip).pk
        name += f"_{other_pk}"

    save_path = os.path.join(job_dir, f"{name}.json")
    return save_path


def report_stoich_method(method_name,
                         method_descrip,
                         other_method_name,
                         other_method_descrip):
    msg = ("Performing linear regression to fit energies from "
           f"{method_name}/{method_descrip}")
    if all([other_method_name is not None,
            other_method_descrip is not None]):
        msg += (f". Fitting to best match results from {other_method_name}/"
                f"{other_method_descrip}")
    print(msg)


def compute_stoich_reg(group_name,
                       method_names,
                       method_descrips,
                       job_dir,
                       other_method_name=None,
                       other_method_descrip=None,
                       other_stoich_name=None,
                       max_num=float('inf')):

    idx = [i for i, name in enumerate(method_names)
           if 'tddft' in name][0]
    method_name = method_names[idx]
    method_descrip = method_descrips[idx]

    print(("Querying database to generate stoich energies from "
           "linear regression..."))

    report_stoich_method(method_name=method_name,
                         method_descrip=method_descrip,
                         other_method_name=other_method_name,
                         other_method_descrip=other_method_descrip)

    save_path = get_stoich_save(job_dir=job_dir,
                                method_name=method_name,
                                method_descrip=method_descrip,
                                other_method_name=other_method_name,
                                other_method_descrip=other_method_descrip)

    if os.path.isfile(save_path):
        print(f"Stoich energies loaded from {save_path}")
        with open(save_path, 'r') as f_open:
            stoich_dict = json.load(f_open)
        return stoich_dict

    x_in, y_out, atoms_list, main_en_dic, geom_dic = get_stoich_data(
        group_name=group_name,
        method_name=method_name,
        method_descrip=method_descrip,
        other_method_name=other_method_name,
        other_method_descrip=other_method_descrip,
        max_num=max_num)

    if all([other_method_name is not None,
            other_method_descrip is not None]):
        stoich_dict = match_stoich(geom_dic=geom_dic,
                                   main_en_dic=main_en_dic,
                                   compare_method_name=other_method_name,
                                   compare_method_descrip=other_method_descrip,
                                   compare_stoich_name=other_stoich_name)

    else:
        stoich_dict = fit_stoich(x_in=x_in,
                                 y_out=y_out,
                                 atoms_list=atoms_list)

    with open(save_path, 'w') as f_open:
        json.dump(stoich_dict, f_open, indent=4)

    return stoich_dict


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


def rm_global_outliers(dset,
                       en_keys,
                       grad_keys,
                       max_std_en,
                       max_std_force,
                       max_val_en,
                       max_val_force,
                       good_idx_dic):

    bad_idx = []
    for key in grad_keys:
        f_to_idx = torch.cat([torch.ones(batch[key].shape[0]) * i for i, batch
                              in enumerate(dset)])
        global_f = torch.cat(dset.props[key])
        mean = global_f.mean()
        std = global_f.std()

        bad_idx += list(set(f_to_idx[(abs(global_f - mean)
                                      > max_val_force).any(-1)
                                     ].tolist()))
        bad_idx += list(set(f_to_idx[(abs(global_f - mean)
                                      > max_std_force * std
                                      ).any(-1)].tolist()))

    for key in en_keys:
        global_en = torch.stack(dset.props[key]).reshape(-1)
        mean = global_en.mean()
        std = global_en.std()

        bad_idx += list(set((abs(global_en - mean) >
                             max_val_en).nonzero().reshape(-1).tolist()))
        bad_idx += list(set((abs(global_en - mean) >
                             max_std_en * std).nonzero().reshape(-1).tolist()))

    bad_idx = list(set(bad_idx))
    for idx in bad_idx:
        good_idx_dic[idx] *= 0

    return good_idx_dic


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

    for sub_dic in en_outlier_dic.values():
        for key in en_keys:
            good_idx_dic = en_trim(good_idx_dic=good_idx_dic,
                                   sub_dic=sub_dic,
                                   key=key,
                                   max_std_en=max_std_en,
                                   max_val_en=max_val_en)

    for sub_dic in grad_outlier_dic.values():
        for key in grad_keys:
            good_idx_dic = grad_trim(good_idx_dic=good_idx_dic,
                                     sub_dic=sub_dic,
                                     key=key,
                                     max_std_force=max_std_force,
                                     max_val_force=max_val_force)

    good_idx_dic = rm_global_outliers(dset=dset,
                                      en_keys=en_keys,
                                      grad_keys=grad_keys,
                                      max_std_en=max_std_en,
                                      max_std_force=max_std_force,
                                      max_val_en=max_val_en,
                                      max_val_force=max_val_force,
                                      good_idx_dic=good_idx_dic)

    final_idx = torch.LongTensor([i for i, val in good_idx_dic.items()
                                  if val == 1])

    for key, val in dset.props.items():
        if isinstance(val, list):
            dset.props[key] = [val[i] for i in final_idx]
        else:
            dset.props[key] = val[final_idx]

    return dset


def add_deltas(dset,
               deltas):
    if deltas is None:
        return dset

    new_props = {}

    for delta_pair in deltas:
        new_key = f"{delta_pair[0]}_{delta_pair[1]}_delta"
        if new_key in dset.props:
            continue

        top_val = dset.props[delta_pair[0]]
        if isinstance(top_val, list):
            delta = [torch.zeros_like(val) for val in top_val]
        else:
            delta = torch.zeros_like(top_val)

        for i, batch in enumerate(dset):
            this_delta = batch[delta_pair[0]] - batch[delta_pair[1]]
            delta[i] = this_delta

        new_props[new_key] = delta

    dset.props.update(new_props)

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
              max_val_force,
              custom_name,
              make_splits=False,
              deltas=None,
              add_ground_en=False):

    # print(len(overall_dict))
    # print({key: len([val for sub_key, val in
    #                  overall_dict.items() if key in val])
    #        for key in required_keys})

    overall_dict = trim_overall(overall_dict, required_keys)
    props = concatenate_dict(*list(overall_dict.values()))

    if add_ground_en:
        props = add_ground(props)

    if custom_name == "mean":
        props = subtract_mean(props)

    dset = Dataset(props=props, units='atomic')

    save_folder = os.path.join(save_dir, str(idx))
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    dset = remove_outliers(dset=dset,
                           max_std_en=max_std_en,
                           max_std_force=max_std_force,
                           max_val_en=max_val_en,
                           max_val_force=max_val_force)

    dset = add_deltas(dset, deltas)

    if make_splits:
        names = ["train", "val", "test"]
        splits = split_train_validation_test(dset,
                                             val_size=val_size,
                                             test_size=test_size,
                                             seed=seed)
    else:
        names = ["dataset"]
        splits = [dset]

    for split, name in zip(splits, names):
        save_path = os.path.join(save_folder, f"{name}.pth.tar")
        split.save(save_path)


def parse_optional(max_geoms,
                   max_geoms_per_dset,
                   molsets,
                   geomsets):
    if max_geoms is None:
        max_geoms = float('inf')
    if molsets is None:
        molsets = []
    if geomsets is None:
        geomsets = []
    if max_geoms_per_dset is None:
        max_geoms_per_dset = max_geoms

    return max_geoms, max_geoms_per_dset, molsets, geomsets


def get_job_pks(method_name,
                method_description,
                molsets,
                geomsets,
                all_calcs_for_geom=False):

    config_names = [key for key, val in CONFIG_DIC.items() if
                    val['name'] == method_name and
                    val['description'] == method_description]

    config_str = ", ".join(config_names)
    if len(config_names) > 1:
        config_str = ", ".join(config_names[:-1])
    if len(config_names) >= 2:
        joiner = 'and' if all_calcs_for_geom else 'or'
        config_str += f" {joiner} {config_names[-1]}"

    print(f"Querying all jobs with config {config_str}...")

    if all_calcs_for_geom:
        job_dic = {}
        for name in config_names:
            jobs = (JobConfig.objects.get(name=name).job_set
                    .filter(status='done'))
            job_dic[name] = jobs

        sort_configs = sorted(config_names, key=lambda x: job_dic[x].count())

        min_job_config = sort_configs[0]
        min_jobs = job_dic[min_job_config]
        geom_pks = min_jobs.values_list('childcalcs__geoms', flat=True)
        geoms = Geom.objects.filter(pk__in=geom_pks)

        job_pks = geoms.values_list('calcs__parentjob', flat=True)
        jobs = Job.objects.filter(pk__in=job_pks,
                                  config__name__in=config_names)

    else:
        job_pks = (JobConfig.objects.filter(name__in=config_names)
                   .values_list('job__pk', flat=True))
        jobs = Job.objects.filter(pk__in=job_pks)

    for molset in molsets:
        jobs = jobs.filter(parentgeom__species__mol__sets__name=molset)
    for geomset in geomsets:
        these_pks = list((GeomSet.objects.get(name=geomset)
                          .geoms.values_list('childjobs', flat=True)))
        these_pks += list((GeomSet.objects.get(name=geomset)
                           .geoms.values_list('parentjob', flat=True)))
        jobs = jobs.filter(pk__in=these_pks)

    job_pks = list(jobs.values_list('pk', flat=True))

    print("Completed query!")

    return job_pks


def get_job_query(job_pks,
                  chunk_size,
                  group_name):

    todo = []
    for _ in range(chunk_size):
        if not job_pks:
            break
        todo += [job_pks.pop()]
    job_query = Job.objects.filter(id__in=todo,
                                   status='done',
                                   group__name=group_name)
    return job_query


def update_stoich(stoich_dict,
                  geom,
                  custom_name,
                  method_names,
                  method_descrips,
                  group_name,
                  atom_dict):

    method_name, method_descrip = get_stoich_method(
        group_name=group_name,
        method_names=method_names,
        method_descriptions=method_descrips,
        custom_name=custom_name)

    formula = geom.stoichiometry.formula
    if formula in stoich_dict:
        return stoich_dict, formula

    if custom_name == "mean":
        stoich_en = 0
    elif custom_name == "lin_reg":
        stoich_en = lin_reg_stoich(formula, atom_dict)
    elif custom_name is not None:
        stoich_en = custom_stoich(formula, custom_name)
    else:
        stoich_en = stoich_energy(formula=formula,
                                  methodname=method_name,
                                  method_description=method_descrip)

    stoich_dict[formula] = stoich_en

    return stoich_dict, formula


def get_stoich_method(group_name,
                      method_names,
                      method_descriptions,
                      custom_name):

    if custom_name is not None:
        return method_names[0], method_descriptions[0]

    project = REFERENCE_PROJECT.get(group_name, DEFAULT_PROJECT)
    group = Group.objects.get(name=project)
    spec = Species.objects.get(group=group,
                               smiles='C')
    has_ref = []

    for name, descrip in zip(method_names,
                             method_descriptions):
        geoms = spec.geom_set.filter(converged=True,
                                     calcs__method__name=name,
                                     calcs__method__description=descrip)
        has_ref.append(bool(geoms))

    if not any(has_ref):
        msg = ("None of the methods have a converged reference geometry "
               "for carbon.")
        raise Exception(msg)

    idx = [i for i, ref in enumerate(has_ref) if ref][0]
    name = method_names[idx]
    descrip = method_descriptions[idx]

    return name, descrip


def subtract_mean(props):
    en_keys = [i for i in props.keys()
               if 'energy' in i and 'grad'
               not in i]

    mean = np.mean([np.mean(props[key])
                    for key in en_keys])
    for key in en_keys:
        for i in range(len(props[key])):
            props[key][i] -= mean

    return props


def add_ground(props):
    en_keys = [i for i in props.keys()
               if 'energy' in i and 'grad' not in i
               and i != 'energy_0']

    for key in en_keys:
        for i in range(len(props[key])):
            props[key][i] *= EV_TO_AU
            props[key][i] += props['energy_0'][i]

    return props


def update_overall(overall_dict,
                   stoich_dict,
                   calc_pk,
                   custom_name,
                   method_names,
                   method_descrips,
                   group_name,
                   atom_dict=None):

    calc = Calc.objects.filter(pk=calc_pk).first()

    if not calc:
        return overall_dict, stoich_dict

    geom = calc.geoms.first()
    geom_id = geom.id
    props = calc.props

    if geom_id not in overall_dict:
        overall_dict[geom_id] = {"geom_id": geom_id,
                                 "nxyz": geom.xyz,
                                 "smiles": geom.species.smiles}

    stoich_dict, formula = update_stoich(stoich_dict=stoich_dict,
                                         geom=geom,
                                         custom_name=custom_name,
                                         method_names=method_names,
                                         method_descrips=method_descrips,
                                         group_name=group_name,
                                         atom_dict=atom_dict)

    stoich_en = stoich_dict[formula]

    if props is None:
        jacobian = calc.jacobian
        if not jacobian:
            return overall_dict, stoich_dict
        forces = jacobian.forces
        overall_dict[geom_id].update({"energy_0_grad":
                                      (-torch.Tensor(forces)).tolist(),
                                      "geom_id": geom_id})
    else:
        if len(props.get('excitedstates', [])) == 0:
            return overall_dict, stoich_dict

        overall_dict[geom_id].update({"energy_0": props['totalenergy']
                                      - stoich_en})

        for i, exc_props in enumerate(props['excitedstates']):
            overall_dict[geom_id].update({f"energy_{i+1}":
                                          exc_props['energy']})

            if 'force_nacv' in exc_props:
                exc_state_num = exc_props['absolutestate']
                conv = 627.5 / 0.529177

                exact_nacv = props['force_nacv'][str(exc_state_num)]

                deriv_nacv_etf = props['deriv_nacv_etf'][str(exc_state_num)]
                gap = exc_props["energy"] - props["totalenergy"]

                # Hartree / bohr -> kcal/mol/A. Need to convert because
                # the conversion in the NFF dataset is unknown for this key
                exact_nacv = (torch.Tensor(exact_nacv) * conv) .tolist()
                approx_nacv = (torch.Tensor(deriv_nacv_etf)
                               * gap * conv).tolist()

                # use the approximate one so that there are no Hellman-Feynman
                # assumptions being made, and instead it's just a scaling used
                # for better training
                overall_dict[geom_id].update({"exact_force_nacv_10": exact_nacv,
                                              "force_nacv_10": approx_nacv})

            if 'forces' in exc_props:
                exc_force = exc_props['forces']
                overall_dict[geom_id].update({f"energy_{i+1}_grad":
                                              (-torch.Tensor(exc_force)).tolist()})

    return overall_dict, stoich_dict


def rm_stereo(smiles):
    new_smiles = smiles.replace("/", "").replace("\\", "")
    return new_smiles


def filter_calcs(calc_pks,
                 geomsets):
    calcs = Calc.objects.filter(pk__in=calc_pks)
    for geomset in geomsets:
        calcs = calcs.filter(geoms__geomsets__name=geomset)
    pks = calcs.values_list('pk', flat=True)
    return pks


def run(group_name,
        method_names,
        method_descriptions,
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
        make_splits=False,
        deltas=None,
        all_calcs_for_geom=False,
        geomsets=None,
        add_ground_en=False,
        job_dir=".",
        other_method_name=None,
        other_method_descrip=None,
        other_stoich_name=None,
        max_stoich_specs=float('inf'),
        **kwargs):

    max_geoms, max_geoms_per_dset, molsets, geomsets = parse_optional(
        max_geoms=max_geoms,
        max_geoms_per_dset=max_geoms_per_dset,
        molsets=molsets,
        geomsets=geomsets)

    job_pks = []
    for name, descrip in zip(method_names, method_descriptions):
        job_pks += get_job_pks(method_name=name,
                               method_description=descrip,
                               all_calcs_for_geom=all_calcs_for_geom,
                               molsets=molsets,
                               geomsets=geomsets)

    ####
    # job_pks = list(Job.objects.filter(
    # parentgeom__parentjob__config__name__contains='bp86',
    #   pk__in=job_pks).values_list('pk', flat=True))
    ####

    geom_count = 0
    i = 0

    overall_dict = {}
    stoich_dict = {}
    atom_dict = None

    if custom_name == "lin_reg":
        atom_dict = compute_stoich_reg(
            group_name=group_name,
            method_names=method_names,
            method_descrips=method_descriptions,
            job_dir=job_dir,
            other_method_name=other_method_name,
            other_method_descrip=other_method_descrip,
            other_stoich_name=other_stoich_name,
            max_num=max_stoich_specs)

    while job_pks:
        print("%d jobs remaining..." % (len(job_pks)))
        sys.stdout.flush()

        job_query = get_job_query(job_pks=job_pks,
                                  chunk_size=chunk_size,
                                  group_name=group_name)

        calc_pks = list(job_query.values_list('childcalcs', flat=True))
        calc_pks = filter_calcs(calc_pks, geomsets)

        for calc_pk in tqdm(calc_pks):

            overall_dict, stoich_dict = update_overall(
                overall_dict=overall_dict,
                stoich_dict=stoich_dict,
                calc_pk=calc_pk,
                custom_name=custom_name,
                method_names=method_names,
                method_descrips=method_descriptions,
                group_name=group_name,
                atom_dict=atom_dict)

            if len(overall_dict) >= max_geoms_per_dset:

                geom_count += len(overall_dict)
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
                          max_val_force=max_val_force,
                          custom_name=custom_name,
                          make_splits=make_splits,
                          deltas=deltas,
                          add_ground_en=add_ground_en)

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
              seed=split_seed,
              max_std_en=max_std_en,
              max_std_force=max_std_force,
              max_val_en=max_val_en,
              max_val_force=max_val_force,
              custom_name=custom_name,
              make_splits=make_splits,
              deltas=deltas,
              add_ground_en=add_ground_en)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str,
                        help=('The path with to the config file'),
                        default='job_info.json')
    args = parser.parse_args()
    config_path = args.config_file
    with open(config_path, 'r') as f:
        kwargs = json.load(f)

    if ('method_name' in kwargs and
            'method_names' not in kwargs):
        kwargs['method_names'] = [kwargs['method_name']]
        kwargs['method_descriptions'] = [kwargs['method_description']]

    try:
        run(**kwargs)
    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem()


if __name__ == "__main__":
    main()
